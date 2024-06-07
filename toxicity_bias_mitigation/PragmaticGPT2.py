from typing import List, Optional, Union, Tuple, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.distributed as dist

from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationConfig
from transformers import GenerationMixin
from transformers.generation.beam_search import BeamSearchScorer, BeamScorer, BeamHypotheses

from transformers.generation.utils import (
    GreedySearchOutput, GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput, 
    SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput,
    BeamSearchOutput, BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput,
    BeamSampleOutput, BeamSampleDecoderOnlyOutput, BeamSampleEncoderDecoderOutput, 
    GenerateOutput)

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
import warnings

class StepwiseOutput:
    def __init__(self, unconditional_probabilities, literal_speaker_probabilities, pragmatic_speaker_probabilities, pragmatic_listener_probabilities, prior_probabilities):
        self.unconditional_probabilities = unconditional_probabilities # bsz * len * vocab
        self.literal_speaker_probabilities = literal_speaker_probabilities # bsz * len * vocab
        self.pragmatic_speaker_probabilities = pragmatic_speaker_probabilities # bsz * len * vocab
        self.pragmatic_listener_probabilities = pragmatic_listener_probabilities # # bsz * len * num_classes * vocab
        self.prior_probabilities = prior_probabilities # bsz * len * num_class: first step uniform

class PragmaticGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, model_classifier, model, alpha, beta, num_classes, prior_aggregation_method="sum", adjustable=False, topk=5):
        self.config = AutoConfig.from_pretrained(model) 
        super().__init__(self.config)        
        self.load_state_dict(GPT2LMHeadModel.from_pretrained(model).state_dict())
        self.eval()
        if model == model_classifier:
            self.has_classifier = False
        else:
            self.classifier_model = AutoModelForCausalLM.from_pretrained(model_classifier)
            self.classifier_model.eval()
            self.has_classifier = True
        torch.set_grad_enabled(False)
        torch.set_default_dtype(torch.float64)
        # left padding for generation
        self.tokenizer_left = AutoTokenizer.from_pretrained(model)
        self.tokenizer_left.padding_side = 'left'
        self.tokenizer_left.pad_token = self.tokenizer_left.eos_token
        # right padding for calculating sentence probabilities
        self.tokenizer_right = AutoTokenizer.from_pretrained(model)
        self.tokenizer_right.pad_token = self.tokenizer_right.eos_token

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

        self.prior_aggregation_method = prior_aggregation_method
        self.adjustable = adjustable
        self.topk = topk
        print('model initialized')

    def use_classifier(self, *args ,**kwargs):
        if self.has_classifier:
            # Use the separate classifier model
            return self.classifier_model(*args ,**kwargs)
        else:
            # Use the main model instance itself
            return self(*args ,**kwargs)  
                
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return

    def prepare_position_ids(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_lengths = attention_mask.sum(dim=1)
        inputs['position_ids'] = torch.zeros(size=attention_mask.shape, dtype=torch.int, device=self.device)

        for i in range(input_lengths.shape[0]):
                
            inputs['position_ids'][i, -input_lengths[i]:] = torch.arange(input_lengths[i], device=self.device)

        return inputs
    
    def prepare_target_distractor_inputs(self, input_texts, target_prompts, distractor_prompts, padding_side):
        """
        input_texts: a list of strings
        target_prompts: a list of prompts that encourage models to produce attributes of interest
        distractor_prompts: a list of prompts that encourage models to produce attributes as distractors
        """
        inputs = []
        inputs += input_texts
        
        for target_prompt in target_prompts:  
            for input_text in input_texts:         
                inputs += [target_prompt + input_text]
        for distractor_prompt in distractor_prompts:
            for input_text in input_texts:         
                inputs += [distractor_prompt + input_text]
        if padding_side == 'left':
            inputs = self.tokenizer_left(inputs, padding=True, truncation=True, return_tensors='pt')
        elif padding_side == 'right':
            inputs = self.tokenizer_right(inputs, padding=True, truncation=True, return_tensors='pt')
        else:
            raise ValueError("Padding direction must be provided: left or right")
        
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()
        
        input_lengths = attention_mask.sum(dim=1)

        real_input_lengths = attention_mask[:input_ids.shape[0]//(self.num_classes+1),:].sum(dim=1).repeat(self.num_classes+1)
        
        length_mask = torch.zeros(size=attention_mask.shape, device=self.device)
        if padding_side == 'left':
        
            for i in range(input_lengths.shape[0]):
                length_mask[i, -real_input_lengths[i]:] = 1
            
        if padding_side == 'right':
            for i in range(input_lengths.shape[0]):
                length_mask[i, input_lengths[i]-real_input_lengths[i]:input_lengths[i]] = 1
            
        
        labels.masked_fill_(length_mask == 0, -100)
        #print("inputs prepared")
        
        return inputs, labels
    
    def compute_prior_distributions(self, 
                                        input_ids: Optional[torch.LongTensor] = None, # contains regular+target+distractors
                                        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                        attention_mask: Optional[torch.FloatTensor] = None,
                                        token_type_ids: Optional[torch.LongTensor] = None,
                                        position_ids: Optional[torch.LongTensor] = None,
                                        head_mask: Optional[torch.FloatTensor] = None,
                                        inputs_embeds: Optional[torch.FloatTensor] = None,
                                        encoder_hidden_states: Optional[torch.Tensor] = None,
                                        encoder_attention_mask: Optional[torch.FloatTensor] = None,
                                        labels: Optional[torch.LongTensor] = None,
                                        use_cache: Optional[bool] = None,
                                        output_attentions: Optional[bool] = None,
                                        output_hidden_states: Optional[bool] = None,
                                        return_dict: Optional[bool] = None,


                                    ):
        
        # inputs must be padded from right

        outputs = self.use_classifier(input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits # batch, length, vocab
        
        

        bsz = logits.shape[0]
        real_bsz = bsz // (self.num_classes+1)
        # calculate normalized probabilities distribution
        loss_fct = CrossEntropyLoss(reduction='none')
        lm_logits = outputs.logits[real_bsz:, ...]
        if labels is not None:
            # move labels to correct device to enable model parallelism
            prompted_labels = labels[real_bsz:, ...].to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = prompted_labels[..., 1:].contiguous()
            # Flatten the tokens
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss_by_sequence = loss.view(lm_logits.size(0), -1) # bsz * len-1
        
        real_loss_mask = (loss_by_sequence!=0.000) # where the real inputs lie, bsz * len
        real_loss_length = real_loss_mask.sum(dim=1) #bsz,

        left_padded_loss_by_sequence = torch.zeros(loss_by_sequence.shape[0], loss_by_sequence.shape[1]).to(self.device)
        for i in range(left_padded_loss_by_sequence.shape[0]):
            #print(real_loss_length[i], real_loss_mask[i, :].sum())
            left_padded_loss_by_sequence[i, -real_loss_length[i]:] = loss_by_sequence[i, real_loss_mask[i, :]]
        
        left_padded_loss_by_sequence = left_padded_loss_by_sequence.cumsum(dim=1)
        if self.prior_aggregation_method == "mean":
            length_mask = torch.ones((loss_by_sequence.shape[0], loss_by_sequence.shape[1]))
            for i in range(real_loss_length.shape[0]):
                length_mask[i,-real_loss_length[i]:] = torch.tensor([j+1 for j in range(real_loss_length[i])])
            
            length_mask = length_mask.to(self.device) 
            left_padded_loss_by_sequence /= length_mask
        
        unnormalized_listener_probability = left_padded_loss_by_sequence.view(self.num_classes, real_bsz, loss_by_sequence.shape[1]).permute(1, 2, 0) # real_bsz * len-1 * num_classes
        
        prior_distributions = F.softmax(-unnormalized_listener_probability, dim=-1) # real_bsz * len-1 * num_classes
        prior_distributions = torch.cat(((torch.ones((real_bsz, 1, self.num_classes))/self.num_classes).to(self.device), prior_distributions), dim=1)

        return prior_distributions


    def pragmatic_modeling(self,
        input_ids: Optional[torch.LongTensor] = None, # contains regular+target+distractors
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        #labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_positive_prompts: int = 1,
        prior_distributions=None,
    ): 
        
        if prior_distributions is None:
            raise ValueError("Missing prior_distributions, please compute the prior distributions using 'compute_prior_distributions' with right padded inputs")
        
        if self.has_classifier:
            outputs = self.use_classifier(input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                #labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            # logits, past_key_values, hidden_states, attentions
            logits = outputs.logits # batch, length, vocab
            prob = logits.softmax(dim=-1)
            bsz = logits.shape[0]
            real_bsz = bsz // (self.num_classes+1)
            other_prob = prob[real_bsz:,:,:] # real_bsz x num_classes, len, vocab

        # generator probability only for updating speaker probability 
        generator_output = self(input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            #labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.has_classifier:
            # logits, past_key_values, hidden_states, attentions
            logits = generator_output.logits # batch, length, vocab
            prob = logits.softmax(dim=-1)
            bsz = logits.shape[0]
            real_bsz = bsz // (self.num_classes+1)
            other_prob = prob[real_bsz:,:,:] # real_bsz x num_classes, len, vocab
        generator_prob = generator_output.logits.softmax(dim=-1)
        regular_prob = generator_prob[:real_bsz,:,:] # real_bsz * len * vocab        
        
        other_prob_by_example = other_prob.view(self.num_classes, real_bsz, logits.shape[1] , logits.shape[-1]).permute(1, 2, 0, 3) # real_bsz * len * num_classes * vocab
        literal_speaker_probabilities = other_prob_by_example[:, :, 0, :]

        pragmatic_listener_probability_distribution = torch.mul(other_prob_by_example, prior_distributions.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])) # real_bsz * len * num_classes * vocab
        pragmatic_listener_probability_distribution = pragmatic_listener_probability_distribution / pragmatic_listener_probability_distribution.sum(dim=2).unsqueeze(dim=2) # real_bsz * len * num_classes * vocab -> prior for next step depending on what token is selected
        mean_positive_distribution = pragmatic_listener_probability_distribution[:,:,:num_positive_prompts,:].sum(dim=2) # real_bsz * len * vocab
        
        pragmatic_speaker_probability_distribution = torch.mul(mean_positive_distribution ** self.alpha, regular_prob)
       
        pragmatic_speaker_probability_distribution = pragmatic_speaker_probability_distribution / pragmatic_speaker_probability_distribution.sum(dim=-1).unsqueeze(dim=-1) # real_bsz * len * vocab
        
        if self.adjustable:
            _, next_step_token_selection_debiased = torch.topk(pragmatic_speaker_probability_distribution, k=self.topk, dim=-1)
            _, next_step_token_selection_default = torch.topk(regular_prob, k=self.topk, dim=-1)
            batch_dim = torch.arange(next_step_token_selection_debiased.size(0)).view(-1, 1, 1).to(next_step_token_selection_debiased.device)
            len_dim = torch.arange(next_step_token_selection_debiased.size(1)).view(1, -1, 1).to(next_step_token_selection_debiased.device)
            # the probability of selecting a non-toxic token
            next_step_non_toxicity_prob_debiased = pragmatic_listener_probability_distribution[batch_dim, len_dim, 0, next_step_token_selection_debiased].mean(dim=-1) # real_bsz * len
            next_step_non_toxicity_prob_default = pragmatic_listener_probability_distribution[batch_dim, len_dim, 0, next_step_token_selection_default].mean(dim=-1) # real_bsz * len
            # the perplexity of next step debiased
            next_step_lm_probability_debiased = regular_prob[batch_dim, len_dim, next_step_token_selection_debiased].mean(dim=-1) # real_bsz * len
            next_step_lm_probability_default = regular_prob[batch_dim, len_dim, next_step_token_selection_default].mean(dim=-1) # real_bsz * len
            # relative ratio for non-toxicity
            non_toxicity_ratio = torch.log(next_step_non_toxicity_prob_debiased) - torch.log(next_step_non_toxicity_prob_default)
            # relative ratio for perplexity
            perplexity_ratio = torch.log(next_step_lm_probability_debiased) - torch.log(next_step_lm_probability_default)
            relative_factor = torch.exp(perplexity_ratio - non_toxicity_ratio)
            relative_factor = torch.min(relative_factor, torch.ones_like(relative_factor).to(relative_factor.device))

            pragmatic_speaker_probability_distribution = torch.mul(mean_positive_distribution ** (self.alpha + self.beta * relative_factor.unsqueeze(-1).repeat(1, 1, regular_prob.shape[-1])), regular_prob)


        return generator_output, StepwiseOutput(regular_prob, literal_speaker_probabilities, pragmatic_speaker_probability_distribution, pragmatic_listener_probability_distribution, prior_distributions)
    
    def classify(self, input_texts, target_prompts, distractor_prompts):
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts, padding_side='right')
        prior_distributions = self.compute_prior_distributions(**inputs, labels=labels)
        pred = prior_distributions[:, -1, :]
        return pred
    
    def debiased_generation(self, input_texts, target_prompts, distractor_prompts, min_length: int = None, max_length: int = None, **kwargs):
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts, padding_side='right')
        prior_distributions = self.compute_prior_distributions(**inputs, labels=labels)
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts, padding_side='left')
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length
        output_ids = self.generate(**inputs, labels=labels, min_length=min_length, max_length=max_length, prior_distributions=prior_distributions, num_positive_prompts=len(target_prompts), use_cache=False, pad_token_id=self.tokenizer_right.eos_token_id, **kwargs)
        if output_ids.shape[0] == inputs['input_ids'].shape[0]: # beam search returns real batch size
            output_ids = output_ids[:(output_ids.shape[0]//(self.num_classes+1)), ...]
        return self.tokenizer_left.batch_decode(output_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    def compute_perplexity(self, input_texts, target_prompts, distractor_prompts):
        assert len(input_texts) == 1, "no batch computation for perplexity"
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts, padding_side='right')       
        prior_distributions = self.compute_prior_distributions(**inputs, labels=labels)
        inputs, labels = self.prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts, padding_side='left')
        outputs, pragmatic_outputs = self.pragmatic_modeling(**inputs, prior_distributions=prior_distributions)
        labels = self.tokenizer_left([self.tokenizer_left.bos_token+input_texts[0]], return_tensors='pt')['input_ids'].clone().to(self.device)
        pragmatic_logits = torch.log(pragmatic_outputs.pragmatic_speaker_probabilities[:, -labels.shape[-1]:, :])
        #print(pragmatic_logits.shape)
        loss_fct = CrossEntropyLoss()

        shift_debiased_logits = pragmatic_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()        
        debiased_loss = loss_fct(shift_debiased_logits.view(-1, shift_debiased_logits.size(-1)), shift_labels.view(-1))

        return debiased_loss
        
 
    def update_prior_distributions(self, input_ids, pragmatic_listener_probabilities, prior_distributions):
        # input_ids: bsz * len+1
        # pragmatic_listener_probabilities: real_bsz * len+1 * num_classes * vocab
        # prior_distributions: real_bsz * len * num_classes
        real_bsz, cur_len, num_classes, vocab = pragmatic_listener_probabilities.shape
        assert num_classes == self.num_classes
        next_token_selection = input_ids[:, -1] # bsz,
        
        # assert torch.cat([next_token_selection[:input_ids.shape[0]//(self.num_classes+1)]]*(self.num_classes+1)) == next_token_selection # make sure for each prompt, a same continuation is selected
        next_token_pragmatic_listener_probabilities = pragmatic_listener_probabilities.permute(2, 0, 1, 3).reshape(real_bsz*num_classes, cur_len, vocab)[:, -1, :] # bsz, vocab
        next_token_pragmatic_listener_probabilities = next_token_pragmatic_listener_probabilities[torch.arange(next_token_pragmatic_listener_probabilities.shape[0]), next_token_selection[input_ids.shape[0]//(self.num_classes+1):]]
        next_token_pragmatic_listener_probabilities = next_token_pragmatic_listener_probabilities.view(real_bsz, num_classes).unsqueeze(1)
        prior_distributions = torch.cat((prior_distributions, next_token_pragmatic_listener_probabilities), dim=1)
        return prior_distributions


#########################################################################################################
#########################################################################################################
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        num_positive_prompts: int = 1,
        prior_distributions=None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs = self.prepare_position_ids(model_inputs)

            # forward pass to get next token
            
            outputs, stepwise_outputs = self.pragmatic_modeling(
                **model_inputs,
                #labels=labels,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                num_positive_prompts=num_positive_prompts,
                prior_distributions=prior_distributions,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])


            # pre-process distribution
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_tokens_scores = logits_processor(real_input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )


            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = torch.stack([next_tokens] * (self.num_classes+1)).reshape(-1,)
            
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            prior_distributions = self.update_prior_distributions(input_ids, stepwise_outputs.pragmatic_listener_probabilities, prior_distributions)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        



    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        num_positive_prompts: int = 1,
        prior_distributions=None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs = self.prepare_position_ids(model_inputs)
            # forward pass to get next token
            outputs, stepwise_outputs = self.pragmatic_modeling(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                num_positive_prompts=num_positive_prompts,
                prior_distributions=prior_distributions
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            # pre-process distribution
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_token_scores = logits_processor(real_input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_tokens = torch.stack([next_tokens] * (self.num_classes+1)).reshape(-1,)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            prior_distributions = self.update_prior_distributions(input_ids, stepwise_outputs.pragmatic_listener_probabilities, prior_distributions)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids



    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        num_positive_prompts: int = 1,
        prior_distributions=None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        ########################################################################

        batch_size = input_ids.shape[0] // ((self.num_classes+1) * num_beams)

        #batch_size = len(beam_scorer._beam_hyps)
        beam_scorer.batch_size = batch_size
        beam_scorer._beam_hyps = [
            BeamHypotheses(
                num_beams=beam_scorer.num_beams,
                max_length=beam_scorer._beam_hyps[0].max_length,
                length_penalty=beam_scorer.length_penalty,
                early_stopping=beam_scorer.do_early_stopping,
            )
            for _ in range(beam_scorer.batch_size)
        ]
        beam_scorer._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=beam_scorer.device)


        ########################################################################


        if num_beams * batch_size != (batch_beam_size // (self.num_classes+1)):
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs = self.prepare_position_ids(model_inputs)
            outputs, stepwise_outputs = self.pragmatic_modeling(
                **model_inputs,
                        
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                num_positive_prompts=num_positive_prompts,
                prior_distributions=prior_distributions,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_token_scores_processed = logits_processor(real_input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                real_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            beam_next_tokens = torch.cat([beam_next_tokens]*(self.num_classes+1)).reshape(-1)
            beam_idx = torch.cat([beam_idx]*(self.num_classes+1)).reshape(-1)            
            beam_idx = beam_idx+torch.tensor([(i//(num_beams*(self.num_classes+1)))*num_beams*self.num_classes for i in range(beam_idx.shape[0])], device=beam_idx.device)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            prior_distributions = self.update_prior_distributions(input_ids, stepwise_outputs.pragmatic_listener_probabilities, prior_distributions)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids[:(input_ids.shape[0]//(self.num_classes+1)), :], 
            beam_scores, 
            next_tokens, 
            next_indices, 
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )
        
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]


    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        num_positive_prompts: int = 1,
        prior_distributions=None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        ########################################################################

        batch_size = input_ids.shape[0] // ((self.num_classes+1) * num_beams)

        #batch_size = len(beam_scorer._beam_hyps)
        beam_scorer.batch_size = batch_size
        beam_scorer._beam_hyps = [
            BeamHypotheses(
                num_beams=beam_scorer.num_beams,
                max_length=beam_scorer._beam_hyps[0].max_length,
                length_penalty=beam_scorer.length_penalty,
                early_stopping=beam_scorer.do_early_stopping,
            )
            for _ in range(beam_scorer.batch_size)
        ]
        beam_scorer._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=beam_scorer.device)


        ########################################################################
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs = self.prepare_position_ids(model_inputs)            
            outputs, stepwise_outputs = self.pragmatic_modeling(
                **model_inputs,
                        
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                num_positive_prompts=num_positive_prompts,
                prior_distributions=prior_distributions,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = torch.log(stepwise_outputs.pragmatic_speaker_probabilities[:, -1, :])

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            real_input_ids = input_ids[:(input_ids.shape[0]//(self.num_classes+1)), ...]
            next_token_scores_processed = logits_processor(real_input_ids, next_token_scores)
            next_token_scores_processed = logits_warper(real_input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                real_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            beam_next_tokens = torch.cat([beam_next_tokens]*(self.num_classes+1)).reshape(-1)
            beam_idx = torch.cat([beam_idx]*(self.num_classes+1)).reshape(-1)            
            beam_idx = beam_idx+torch.tensor([(i//(num_beams*(self.num_classes+1)))*num_beams*self.num_classes for i in range(beam_idx.shape[0])], device=beam_idx.device)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            prior_distributions = self.update_prior_distributions(input_ids, stepwise_outputs.pragmatic_listener_probabilities, prior_distributions)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids[:(input_ids.shape[0]//(self.num_classes+1)), :], 
            beam_scores, 
            next_tokens, 
            next_indices, 
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]
