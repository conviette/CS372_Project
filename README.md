# CS372_Project

## Data Directory

Most data files (text corpus, economic indicators, keyword files, feature files) are saved here.
We currently have three text corpora and three economic indicators.
### Text
* Inaugural (in NLTK, only once in four years)
* SOTUS (State of the Union Speech, once every year)
* Oral (All Oral speeches, all speeches in the same year are concatenated to one corpus)
### Indicators
* GDP growth rate
*  Export Index
* Unemployment Rate

Indicators are from 1961 to 2018 (Export is from 1980 to 2018). Because of this, we don't have a lot of data points to work with, **especially if you use the inaugural corpus**. Please keep this in mind!

## Code Directory

### For keyword generation
* keyword_group.py: Gets groups of keywords from wordnet. Output is keyword_group.json in the Data directory.
* keyword_from_reuters.py: Gets group of keywords from Reuters news dataset. Output is reuters_keywords.json in the Data Directory.

If you decide to start a new approach to finding keywords, please do the following:

 1. Create new py file in Code directory.
 2. Make it create a txt or json (your choice) file in the Data directory.
 3. Add a function to calc_features.py that opens the txt/json file and turns the keywords into a dictionary format. (group name:[list of keywords] format)

### For feature generation
* calc_features.py: File for computing features for text. Output is name_scores.json file in the Data directory.
* project_dataset.py: Module file for reading text data.

If you decide to add features, please do the following:

 1. inside calc_features, please define a function that takes the corpus (list of sentences) and the feature_dict (dictionary of feature_names:feature) as input. (It can take other things as inputs too, such as total_len)
 2. The function should modify the feature_dict by adding your feature name and feature as key/value.
 3. Go to the function find_features(corpus_dict) and execute your function with the corpus and feature_dict as input.

### For linear regression
* lr.py: linear regression using statsmodel library. prints the p-value and R^2 values, also saves all information to a txt file in Results. At the top you can change the STAT_NAME and CORPUS_NAME variable to test for other indicators and corpus.

I think we should keep this part simple as this part is mostly unrelated to our course. This is just to score the performance of the above steps.
A smaller p-value (0 - 1, preferably <0.05) and a larger R^2 value (0 - 1, closer to 1) is desirable.
