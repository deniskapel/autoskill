from sklearn.preprocessing import MultiLabelBinarizer

# AllMidas2ID = {
#     "appreciation": 0, "command": 1, "comment": 2,"complaint": 3,
#     "dev_command": 4, "neg_answer": 5, "open_question_factual": 6,
#     "open_question_opinion": 7, "opinion": 8, "other_answers": 9,
#     "pos_answer": 10, "statement": 11, "yes_no_question": 12,
# }
#
# ID2AllMidas = list(AllMidas2ID.keys())

Midas2ID = {
    "command": 0, "comment": 1, "neg_answer": 2,
    "open_question_factual": 3, "open_question_opinion": 4,
    "opinion": 5, "pos_answer": 6, "statement": 7, "yes_no_question": 8,
}

ID2Midas = list(Midas2ID.keys())


Entity2ID = {'misc': 0, 'product': 1, 'food': 2, 'location': 3, 'business': 4,
             'event': 5, 'work_of_art': 6, 'org': 7, 'occupation': 8, 'fac': 9,
             'academic_discipline': 10, 'law': 11, 'film': 12, 'person': 13,
             'language': 14, 'type_of_sport': 15, 'nation': 16, 'literary_work': 17,
             'norp': 18, 'music_genre': 19, 'sports_event': 20, 'song': 21,
             'animal': 22, 'sports_venue': 23, 'sports_season': 24,
             'chemical_element': 25, 'political_party': 26, 'sport_team': 27,
             'national': 28, 'championship': 29, 'association_football_club': 30,
             'sports_league': 31}


EntityTargets2ID = {'product': 0, 'food': 1, 'location': 2, 'business': 3,
                    'event': 4, 'work_of_art': 5, 'org': 6, 'occupation': 7,
                    'fac': 8, 'academic_discipline': 9, 'law': 10, 'person': 11,
                    'language': 12, 'type_of_sport': 13, 'nation': 14,
                    'norp': 15, 'music_genre': 16, 'sports_event': 17,
                    'animal': 18, 'sports_venue': 19, 'sports_season': 20,
                    'chemical_element': 21, 'political_party': 22,
                    'sport_team': 23, 'national': 24, 'championship': 25,
                    'association_football_club': 26, 'sports_league': 27}


EntityLabelEncoder = MultiLabelBinarizer()
EntityLabelEncoder.classes = [label for label in EntityTargets2ID]
ID2Entity = EntityLabelEncoder.classes

all_labels = [label for label in ID2Midas + [None] + EntityLabelEncoder.classes]
