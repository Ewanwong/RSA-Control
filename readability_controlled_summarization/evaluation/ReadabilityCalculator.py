import textstat

def calc_readability(input_data):
    #  flesch_kincaid_grade gunning_fog SMOG automated_readability_index coleman_liau_index
    if not input_data:
        raise ValueError("Input data cannot be empty.")

    prediction_flesch_kincaid_grade_list = []
    reference_flesch_kincaid_grade_list = []

    prediction_gunning_fog_list = []
    reference_gunning_fog_list = []

    prediction_smog_list = []
    reference_smog_list = []

    prediction_automated_readability_index_list = []
    reference_automated_readability_index_list = []

    prediction_coleman_liau_index_list = []
    reference_coleman_liau_index_list = []
    
    prediction_flesch_reading_ease_list = []
    reference_flesch_reading_ease_list = []

    prediction_dale_chall_readability_score_list = []
    reference_dale_chall_readability_score_list = []

    prediction_difficult_words_list = []
    reference_difficult_words_list = []


    prediction_linsear_write_formula_list = []
    reference_linsear_write_formula_list = []

    prediction_text_standard_list = []
    reference_text_standard_list = []

    prediction_spache_readability_list = []
    reference_spache_readability_list = []

    for pair in input_data:

        try:
            reference = pair['reference'].strip()
            prediction = pair['prediction'].strip()
        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
        
        prediction_difficult_words_list.append(textstat.difficult_words(prediction))
        prediction_dale_chall_readability_score_list.append(textstat.dale_chall_readability_score(prediction))
        prediction_flesch_reading_ease_list.append(textstat.flesch_reading_ease(prediction))
        prediction_flesch_kincaid_grade_list.append(textstat.flesch_kincaid_grade(prediction))
        prediction_gunning_fog_list.append(textstat.gunning_fog(prediction))
        prediction_smog_list.append(textstat.smog_index(prediction))
        prediction_automated_readability_index_list.append(textstat.automated_readability_index(prediction))
        prediction_coleman_liau_index_list.append(textstat.coleman_liau_index(prediction))
        prediction_linsear_write_formula_list.append(textstat.linsear_write_formula(prediction))
        prediction_text_standard_list.append(textstat.text_standard(prediction, float_output=True))
        prediction_spache_readability_list.append(textstat.spache_readability(prediction))


        reference_difficult_words_list.append(textstat.difficult_words(reference))
        reference_dale_chall_readability_score_list.append(textstat.dale_chall_readability_score(reference))
        reference_flesch_reading_ease_list.append(textstat.flesch_reading_ease(reference))
        reference_flesch_kincaid_grade_list.append(textstat.flesch_kincaid_grade(reference))
        reference_gunning_fog_list.append(textstat.gunning_fog(reference))
        reference_smog_list.append(textstat.smog_index(reference))
        reference_automated_readability_index_list.append(textstat.automated_readability_index(reference))
        reference_coleman_liau_index_list.append(textstat.coleman_liau_index(reference))
        reference_linsear_write_formula_list.append(textstat.linsear_write_formula(reference))
        reference_text_standard_list.append(textstat.text_standard(reference, float_output=True))
        reference_spache_readability_list.append(textstat.spache_readability(reference))

    
    doc_num = len(input_data)
    return [{"prediction_flesch_kincaid_grade":round(sum(prediction_flesch_kincaid_grade_list)/doc_num, 4),"reference_flesch_kincaid_grade":round(sum(reference_flesch_kincaid_grade_list)/doc_num, 4)},
            {"prediction_gunning_fog":round(sum(prediction_gunning_fog_list)/doc_num, 4),"reference_gunning_fog":round(sum(reference_gunning_fog_list)/doc_num, 4)},
            {"prediction_smog":round(sum(prediction_smog_list)/doc_num, 4),"reference_smog":round(sum(reference_smog_list)/doc_num, 4)},
            {"prediction_automated_readability_index":round(sum(prediction_automated_readability_index_list)/doc_num, 4),"reference_automated_readability_index":round(sum(reference_automated_readability_index_list)/doc_num, 4)},
            {"prediction_coleman_liau_index":round(sum(prediction_coleman_liau_index_list)/doc_num, 4),"reference_coleman_liau_index":round(sum(reference_coleman_liau_index_list)/doc_num, 4)},
            {"prediction_flesch_reading_ease":round(sum(prediction_flesch_reading_ease_list)/doc_num, 4),"reference_flesch_reading_ease":round(sum(reference_flesch_reading_ease_list)/doc_num, 4)},
            {"prediction_dale_chall_readability_score":round(sum(prediction_dale_chall_readability_score_list)/doc_num, 4),"reference_dale_chall_readability_score":round(sum(reference_dale_chall_readability_score_list)/doc_num, 4)},
            {"prediction_difficult_words":round(sum(prediction_difficult_words_list)/doc_num, 4),"reference_difficult_words":round(sum(reference_difficult_words_list)/doc_num, 4)},
            {"prediction_linsear_write_formula":round(sum(prediction_linsear_write_formula_list)/doc_num, 4),"reference_linsear_write_formula":round(sum(reference_linsear_write_formula_list)/doc_num, 4)},
            {"prediction_text_standard":round(sum(prediction_text_standard_list)/doc_num, 4),"reference_text_standard":round(sum(reference_text_standard_list)/doc_num, 4)},
            {"prediction_spache_readability":round(sum(prediction_spache_readability_list)/doc_num, 4),"reference_spache_readability":round(sum(reference_spache_readability_list)/doc_num, 4)}]
    


if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_path', required=True, type=str, help='Path to the input JSON file')
    args = parser.parse_args()

    output_json_path = args.input_json_path.replace(".json", "_readability.json")
    with open(args.input_json_path, 'r') as f:
        data=json.load(f)



    with open(output_json_path, 'w') as f:
        
        data_readability_stat = calc_readability(data)
        data = json.dumps(data_readability_stat)
        data = data.replace("}, ", "},\n")
        f.write(data)