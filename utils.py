from typing import List

import matplotlib.pyplot as plt
import numpy
import pandas
from joblib import load
from scipy.stats import halfnorm
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from clean_analyzing_bios import tokenize

# Probability dictionary
p = {}

# Movie Genres
movies = ['Adventure',
          'Action',
          'Drama',
          'Comedy',
          'Thriller',
          'Horror',
          'RomCom',
          'Musical',
          'Documentary']

p['Movies'] = [0.28,
               0.21,
               0.16,
               0.14,
               0.09,
               0.06,
               0.04,
               0.01,
               0.01]

# TV Genres
tv = ['Comedy',
      'Drama',
      'Action/Adventure',
      'Suspense/Thriller',
      'Documentaries',
      'Crime/Mystery',
      'News',
      'SciFi',
      'History']

p['TV'] = [0.30,
           0.23,
           0.12,
           0.12,
           0.09,
           0.08,
           0.03,
           0.02,
           0.01]

# Religions (could potentially create a spectrum)
religion = ['Catholic',
            'Christian',
            'Jewish',
            'Muslim',
            'Hindu',
            'Buddhist',
            'Spiritual',
            'Other',
            'Agnostic',
            'Atheist']

p['Religion'] = [0.16,
                 0.16,
                 0.01,
                 0.19,
                 0.11,
                 0.05,
                 0.10,
                 0.09,
                 0.07,
                 0.06]

# Music
music = ['Rock',
         'HipHop',
         'Pop',
         'Country',
         'Latin',
         'EDM',
         'Gospel',
         'Jazz',
         'Classical']

p['Music'] = [0.30,
              0.23,
              0.20,
              0.10,
              0.06,
              0.04,
              0.03,
              0.02,
              0.02]

# Sports
sports = ['Football',
          'Baseball',
          'Basketball',
          'Hockey',
          'Soccer',
          'Other']

p['Sports'] = [0.34,
               0.30,
               0.16,
               0.13,
               0.04,
               0.03]

# Politics (could also put on a spectrum)
politics = ['Liberal',
            'Progressive',
            'Centrist',
            'Moderate',
            'Conservative']

p['Politics'] = [0.26,
                 0.11,
                 0.11,
                 0.15,
                 0.37]

# Social Media
social = ['Facebook',
          'Youtube',
          'Twitter',
          'Reddit',
          'Instagram',
          'Pinterest',
          'LinkedIn',
          'SnapChat',
          'TikTok']

p['Social Media'] = [0.36,
                     0.27,
                     0.11,
                     0.09,
                     0.05,
                     0.03,
                     0.03,
                     0.03,
                     0.03]

age = None

# Lists of Names and the list of the lists
categories = [movies, tv, religion, music, politics, social, sports, age]

names = ['Movies', 'TV', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Age']

combined = dict(zip(names, categories))


def vectorized_words_count_vector(raw_document, is_bigrams: bool = False):
    # Instantiating the Vectorizer
    if is_bigrams:
        vectorizer = CountVectorizer(ngram_range=(2, 2))
    else:
        vectorizer = CountVectorizer()

    # Fitting the vectorizer to the Bios
    vector = vectorizer.fit_transform(raw_document)
    return pandas.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())


def vectorized_words_tfidf(raw_document, is_bigrams: bool = False):
    # Instantiating the Vectorizer
    if is_bigrams:
        vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    else:
        vectorizer = TfidfVectorizer()

    # Fitting the vectorizer to the Bios
    vector = vectorizer.fit_transform(raw_document)
    return pandas.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())


def cluster_eval(y, x):
    """
    Plots the scores of a set evaluation metric. Prints out the max and min values of the evaluation scores.
    """

    # Creating a DataFrame for returning the max and min scores for each cluster
    df = pandas.DataFrame(columns=['Cluster Score'], index=[i for i in range(2, len(y) + 2)])
    df['Cluster Score'] = y

    print('-' * 20)
    print('Max Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].max()])
    print('\nMin Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].min()])
    print('-' * 20)

    # Plotting out the scores based on cluster count
    plt.figure(figsize=(16, 6))
    plt.style.use('ggplot')
    plt.plot(x, y)
    plt.xlabel('# of Clusters')
    plt.ylabel('Score')
    plt.show()


def pca_data_frame_99(data_frame):
    from sklearn.decomposition import PCA

    # Instantiating PCA
    pca = PCA()

    # Fitting and Transforming the DF
    df_pca = pca.fit_transform(data_frame)

    # Plotting to determine how many features should the dataset be reduced to
    plt.style.use("bmh")
    plt.figure(figsize=(14, 4))
    plt.plot(range(1, data_frame.shape[1] + 1), pca.explained_variance_ratio_.cumsum())
    plt.show()

    # Finding the exact number of features that explain at least 99% of the variance in the dataset
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_99 = len(total_explained_variance[total_explained_variance >= .99])
    n_to_reach_99 = data_frame.shape[1] - n_over_99

    print(
        f"Number features: {n_to_reach_99}\nTotal Variance Explained: {total_explained_variance[n_to_reach_99]}")

    # Reducing the dataset to the number of features determined before
    pca = PCA(n_components=n_to_reach_99)

    # Fitting and transforming the dataset to the stated number of features
    df_pca = pca.fit_transform(data_frame)

    # Seeing the variance ratio that still remains after the dataset has been reduced
    print(f"Seeing the variance ratio that still remains after the dataset has been reduced :"
          f"{pca.explained_variance_ratio_.cumsum()[-1]}")

    return df_pca


def pca_data_frame_95(data_frame):
    # Instantiating PCA
    pca = PCA()

    # Fitting and Transforming the DF
    df_pca = pca.fit_transform(data_frame)

    # Plotting to determine how many features should the dataset be reduced to
    plt.style.use("bmh")
    plt.figure(figsize=(14, 4))
    plt.plot(range(1, data_frame.shape[1] + 1), pca.explained_variance_ratio_.cumsum())
    plt.show()

    # Finding the exact number of features that explain at least 95% of the variance in the dataset
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_95 = len(total_explained_variance[total_explained_variance >= .95])
    n_to_reach_95 = data_frame.shape[1] - n_over_95

    # Printing out the number of features needed to retain 95% variance
    print(
        f"Number features: {n_to_reach_95}\nTotal Variance Explained: {total_explained_variance[n_to_reach_95]}")

    # Reducing the dataset to the number of features determined before
    pca = PCA(n_components=n_to_reach_95)

    # Fitting and transforming the dataset to the stated number of features and creating a new DF
    df_pca = pca.fit_transform(data_frame)

    # Seeing the variance ratio that still remains after the dataset has been reduced
    print(pca.explained_variance_ratio_.cumsum()[-1])
    return df_pca


def scaling_categories(data_frame):
    scaler = MinMaxScaler()

    # Scaling the categories then replacing the old values
    return data_frame[['Bios']].join(
        pandas.DataFrame(scaler.fit_transform(data_frame.drop('Bios', axis=1)), columns=data_frame.columns[
                                                                                        1:],
                         index=data_frame.index))


def agglomerative_clustering(data_frame, n_clusters, **kwargs):
    # Instantiating HAC
    hac = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)

    # Fitting
    hac.fit(data_frame)

    # Getting cluster assignments
    return hac.labels_


def k_means_clustering(data_frame, n_clusters, **kwargs):
    # Clustering with different number of clusters
    k_means = KMeans(n_clusters=n_clusters, **kwargs)

    k_means.fit(data_frame)

    return k_means.predict(data_frame)


def mean_shift_clustering(data_frame):
    # define the model
    model = MeanShift(bandwidth=6)
    model.fit(data_frame)
    return model.fit_predict(data_frame)


def refining_profile_data(df):
    # Removing the numerical data
    df = df[['Bios']]

    # Creating Lists for the Categories

    # Probability dictionary
    p = {}

    # Movie Genres
    movies = ['Adventure',
              'Action',
              'Drama',
              'Comedy',
              'Thriller',
              'Horror',
              'RomCom',
              'Musical',
              'Documentary']

    p['Movies'] = [0.28,
                   0.21,
                   0.16,
                   0.14,
                   0.09,
                   0.06,
                   0.04,
                   0.01,
                   0.01]

    # TV Genres
    tv = ['Comedy',
          'Drama',
          'Action/Adventure',
          'Suspense/Thriller',
          'Documentaries',
          'Crime/Mystery',
          'News',
          'SciFi',
          'History']

    p['TV'] = [0.30,
               0.23,
               0.12,
               0.12,
               0.09,
               0.08,
               0.03,
               0.02,
               0.01]

    # Religions (could potentially create a spectrum)
    religion = ['Catholic',
                'Christian',
                'Jewish',
                'Muslim',
                'Hindu',
                'Buddhist',
                'Spiritual',
                'Other',
                'Agnostic',
                'Atheist']

    p['Religion'] = [0.16,
                     0.16,
                     0.01,
                     0.19,
                     0.11,
                     0.05,
                     0.10,
                     0.09,
                     0.07,
                     0.06]

    # Music
    music = ['Rock',
             'HipHop',
             'Pop',
             'Country',
             'Latin',
             'EDM',
             'Gospel',
             'Jazz',
             'Classical']

    p['Music'] = [0.30,
                  0.23,
                  0.20,
                  0.10,
                  0.06,
                  0.04,
                  0.03,
                  0.02,
                  0.02]

    # Sports
    sports = ['Football',
              'Baseball',
              'Basketball',
              'Hockey',
              'Soccer',
              'Other']

    p['Sports'] = [0.34,
                   0.30,
                   0.16,
                   0.13,
                   0.04,
                   0.03]

    # Politics (could also put on a spectrum)
    politics = ['Liberal',
                'Progressive',
                'Centrist',
                'Moderate',
                'Conservative']

    p['Politics'] = [0.26,
                     0.11,
                     0.11,
                     0.15,
                     0.37]

    # Social Media
    social = ['Facebook',
              'Youtube',
              'Twitter',
              'Reddit',
              'Instagram',
              'Pinterest',
              'LinkedIn',
              'SnapChat',
              'TikTok']

    p['Social Media'] = [0.36,
                         0.27,
                         0.11,
                         0.09,
                         0.05,
                         0.03,
                         0.03,
                         0.03,
                         0.03]

    # Age (generating random numbers based on half normal distribution)
    age = halfnorm.rvs(loc=18, scale=8, size=df.shape[0]).astype(int)

    # Lists of Names and the list of the lists
    categories = [movies, tv, religion, music, politics, social, sports, age]

    names = ['Movies', 'TV', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports', 'Age']

    combined = dict(zip(names, categories))

    # Establishing random values for each category

    # Looping through and assigning random values
    for name, cats in combined.items():
        if name in ['Religion', 'Politics']:
            # Picking only 1 from the list
            df[name] = numpy.random.choice(cats, df.shape[0], p=p[name])

        elif name == 'Age':
            # Generating random ages based on a normal distribution
            df[name] = cats
        else:
            # Picking 3 from the list
            try:
                df[name] = list(numpy.random.choice(cats, size=(df.shape[0], 1, 3), p=p[name]))
            except Exception as ex:
                print(ex)
                df[name] = list(numpy.random.choice(cats, size=(df.shape[0], 1, 3)))

            df[name] = df[name].apply(lambda x: list(set(x[0].tolist())))

    df['Religion'] = pandas.Categorical(df.Religion, ordered=True,
                                        categories=['Catholic',
                                                    'Christian',
                                                    'Jewish',
                                                    'Muslim',
                                                    'Hindu',
                                                    'Buddhist',
                                                    'Spiritual',
                                                    'Other',
                                                    'Agnostic',
                                                    'Atheist'])

    df['Politics'] = pandas.Categorical(df.Politics, ordered=True,
                                        categories=['Liberal',
                                                    'Progressive',
                                                    'Centrist',
                                                    'Moderate',
                                                    'Conservative'])

    return df


def clustering(data_frame, fn_vectorized_words, fn_algorithm_clustering, n_clusters,
               is_bigrams: bool = False):
    # Applying the function to each user bio
    data_frame['Bios'] = data_frame.Bios.apply(tokenize)

    scale_df = scaling_categories(data_frame)

    # Creating a new DF that contains the vectorized words
    df_words = fn_vectorized_words(scale_df['Bios'], is_bigrams)

    # Concating the words DF with the original DF
    new_df = pandas.concat([scale_df, df_words], axis=1)

    # Dropping the Bios because it is no longer needed in place of vectorization
    new_df.drop('Bios', axis=1, inplace=True)

    # Fitting and transforming the dataset to the stated number of features
    if is_bigrams:
        df_pca = pca_data_frame_99(new_df)
    else:
        df_pca = pca_data_frame_95(new_df)

    cluster_assignments = fn_algorithm_clustering(df_pca, n_clusters)

    data_frame["Cluster #"] = cluster_assignments
    new_df["Cluster #"] = cluster_assignments
    return data_frame, new_df


def finding_the_right_number_of_clusters(data_frame, fn_vectorized_words,
                                         fn_algorithm_clustering, is_bigrams: bool = False):
    # Applying the function to each user bio
    data_frame['Bios'] = data_frame.Bios.apply(tokenize)

    scale_df = scaling_categories(data_frame)

    # Creating a new DF that contains the vectorized words
    df_words = fn_vectorized_words(scale_df['Bios'], is_bigrams)

    # Concating the words DF with the original DF
    new_df = pandas.concat([scale_df, df_words], axis=1)

    # Dropping the Bios because it is no longer needed in place of vectorization
    new_df.drop('Bios', axis=1, inplace=True)

    # Fitting and transforming the dataset to the stated number of features
    if is_bigrams:
        df_pca = pca_data_frame_99(new_df)
    else:
        df_pca = pca_data_frame_95(new_df)

    # Finding the Optimum Number of Clusters
    # Setting the amount of clusters to test out
    cluster_cnt = [i for i in range(5, 50, 5)] if is_bigrams else [i for i in range(2, 20, 1)]

    # Establishing empty lists to store the scores for the evaluation metrics
    ch_scores = []

    s_scores = []

    db_scores = []

    # Looping through different iterations for the number of clusters
    for i in cluster_cnt:
        # Clustering with different number of clusters
        cluster_assignments = fn_algorithm_clustering(df_pca, i)

        # Appending the scores to the empty lists
        ch_scores.append(calinski_harabasz_score(df_pca, cluster_assignments))

        s_scores.append(silhouette_score(df_pca, cluster_assignments))

        db_scores.append(davies_bouldin_score(df_pca, cluster_assignments))

    # Calinski-Harabasz - A higher scores means better defined clusters.  Aiming for a high score
    cluster_eval(ch_scores, cluster_cnt)

    # Silhouette Coefficient - A higher score means better defined clusters. Aim for high score.
    cluster_eval(s_scores, cluster_cnt)

    # Davies-Bouldin - A lower score is better.  Scores closer to zero are better.
    cluster_eval(db_scores, cluster_cnt)

    k = find_best_cluster(cluster_cnt, ch_scores, s_scores, db_scores)
    print(f"find_best_cluster: {k}")

    cluster_assignments = fn_algorithm_clustering(df_pca, k)

    data_frame["Cluster #"] = cluster_assignments
    new_df["Cluster #"] = cluster_assignments
    return data_frame, new_df


def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x


def clustering_refined_data(data_frame, fn_algorithm_clustering, n_clusters):
    # Applying the function to each user bio
    data_frame['Bios'] = data_frame.Bios.apply(tokenize)

    # Looping through the columns and applying the function
    for col in data_frame.columns:
        data_frame[col] = data_frame[col].apply(string_convert)

    # Creating the vectorized DF
    vect_df = vectorization(data_frame, data_frame.columns)

    scaler = MinMaxScaler()

    vect_df = pandas.DataFrame(scaler.fit_transform(vect_df), index=vect_df.index, columns=vect_df.columns)

    df_pca = pca_data_frame_99(vect_df)

    cluster_assignments = fn_algorithm_clustering(df_pca, n_clusters, linkage='complete')

    data_frame["Cluster #"] = cluster_assignments
    vect_df['Cluster #'] = cluster_assignments
    return data_frame, vect_df


def finding_number_of_clusters_refined_data(data_frame, fn_algorithm_clustering):
    df = data_frame.copy()
    # Applying the function to each user bio
    df['Bios'] = df.Bios.apply(tokenize)

    # Looping through the columns and applying the function
    for col in df.columns:
        df[col] = df[col].apply(string_convert)

    # Creating the vectorized DF
    vect_df = vectorization(df, df.columns)

    scaler = MinMaxScaler()

    vect_df = pandas.DataFrame(scaler.fit_transform(vect_df), index=vect_df.index, columns=vect_df.columns)

    # Instantiating PCA
    pca = PCA()

    # Fitting and Transforming the DF
    df_pca = pca.fit_transform(vect_df)

    # Finding the exact number of features that explain at least 99% of the variance in the dataset
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_9 = len(total_explained_variance[total_explained_variance >= .99])
    n_to_reach_9 = vect_df.shape[1] - n_over_9

    print("PCA reduces the # of features from", vect_df.shape[1], 'to', n_to_reach_9)

    # Reducing the dataset to the number of features determined before
    pca = PCA(n_components=n_to_reach_9)

    # Fitting and transforming the dataset to the stated number of features
    df_pca = pca.fit_transform(vect_df)

    # Seeing the variance ratio that still remains after the dataset has been reduced
    print(pca.explained_variance_ratio_.cumsum()[-1])

    # Setting the amount of clusters to test out
    cluster_cnt = [i for i in range(2, 20, 1)]

    # Establishing empty lists to store the scores for the evaluation metrics
    ch_scores = []

    s_scores = []

    db_scores = []

    # The DF for evaluation
    eval_df = df_pca

    # Looping through different iterations for the number of clusters
    for i in cluster_cnt:
        # Clustering with different number of clusters
        cluster_assignments = fn_algorithm_clustering(eval_df, i, linkage='complete')

        # Appending the scores to the empty lists
        ch_scores.append(calinski_harabasz_score(eval_df, cluster_assignments))

        s_scores.append(silhouette_score(eval_df, cluster_assignments))

        db_scores.append(davies_bouldin_score(eval_df, cluster_assignments))

    print("\nThe Calinski-Harabasz Score (find max score):")
    cluster_eval(ch_scores, cluster_cnt)

    print("\nThe Silhouette Coefficient Score (find max score):")
    cluster_eval(s_scores, cluster_cnt)

    print("\nThe Davies-Bouldin Score (find minimum score):")
    cluster_eval(db_scores, cluster_cnt)

    k = find_best_cluster(cluster_cnt, ch_scores, s_scores, db_scores)
    print(f"find_best_cluster {k}")

    cluster_assignments = fn_algorithm_clustering(df_pca, k)

    data_frame["Cluster #"] = cluster_assignments
    vect_df['Cluster #'] = cluster_assignments
    return data_frame, vect_df


def find_list_profile_by_cluster(data_frame, Vectorizer, k_cluster):
    # Assigning the Cluster Profiles as a new DF
    group = data_frame[data_frame['Cluster #'] == k_cluster].drop('Cluster #', axis=1)

    # Vectorizing the Bios in the Selected Cluster
    vectorizer = Vectorizer()

    # Fitting the vectorizer to the Bios
    cluster_x = vectorizer.fit_transform(group['Bios'])

    # Creating a new DF that contains the vectorized words
    cluster_v = pandas.DataFrame(cluster_x.toarray(), index=group.index,
                                 columns=vectorizer.get_feature_names())

    # Joining the vector DF and the original DF
    group = group.join(cluster_v)

    # Dropping the Bios because it is no longer needed in place of vectorization
    group.drop('Bios', axis=1, inplace=True)

    # Finding Correlations among the users

    # Trasnposing the DF so that we are correlating with the index(users)
    corr_group = group.T.corr()
    return corr_group


def get_similar_profile(data_frame, Vectorizer, k_cluster, limit):
    corr_group = find_list_profile_by_cluster(data_frame, Vectorizer, k_cluster)


def find_best_model_classification_of_new_profile(vector_df):
    # Assigning the split variables
    x = vector_df.drop(["Cluster #"], 1)
    y = vector_df['Cluster #']

    # Train, test, split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    """
    ### Finding the Best Model
    - Dummy (Baseline Model)
    - KNN
    - SVM
    - NaiveBayes
    - Logistic Regression
    - Adaboost
    """

    # Dummy
    dummy = DummyClassifier(strategy='stratified')

    # KNN
    knn = KNeighborsClassifier()

    # SVM
    svm = SVC(gamma='scale')

    # NaiveBayes
    nb = ComplementNB()

    # Logistic Regression
    lr = LogisticRegression()

    # Adaboost
    adab = AdaBoostClassifier()

    # List of models
    models = [dummy, knn, svm, nb, lr, adab]

    # List of model names
    names = ['Dummy', 'KNN', 'SVM', 'NaiveBayes', 'Logistic Regression', 'Adaboost']

    # Zipping the lists
    classifiers = dict(zip(names, models))

    # Visualization of the different cluster counts
    vector_df['Cluster #'].value_counts().plot(kind='pie', title='Count of Class Distribution')

    """Since we are dealing with an imbalanced dataset _(because each cluster is not guaranteed to have the 
    same amount of profiles)_, we will resort to using the __Macro Avg__ and __F1 Score__ for evaluating 
    the performances of each model. """

    # Dictionary containing the model names and their scores
    models_f1 = {}

    # Looping through each model's predictions and getting their classification reports
    for name, model in classifiers.items():
        # Fitting the model
        model.fit(x_train, y_train)

        print('\n' + name + ' (Macro Avg - F1 Score):')

        # Classification Report
        report = classification_report(y_test, model.predict(x_test), output_dict=True)
        f1 = report['macro avg']['f1-score']

        # Assigning to the Dictionary
        models_f1[name] = f1

        print(f1)

    # Model with the Best Performance
    print(max(models_f1, key=models_f1.get), 'Score:', max(models_f1.values()))

    # Fitting the Best Model to our Dataset
    # Fitting the model
    best_model = classifiers[max(models_f1, key=models_f1.get)]
    best_model.fit(x, y)

    return best_model


def find_best_cluster(cluster_cnt, ch_scores, s_scores, db_scores):
    ch_rank = rankify(ch_scores)
    print("ch_scores")
    print(ch_scores)
    print(ch_rank)
    s_rank = rankify(s_scores)
    print("s_scores")
    print(s_scores)
    print(s_rank)
    db_rank = rankify(db_scores, is_revers=True)
    print("db_scores")
    print(db_scores)
    print(db_rank)
    rank = [0 for i in range(len(cluster_cnt))]
    for i in range(len(cluster_cnt)):
        rank[i] = ch_rank[i] + s_rank[i] + db_rank[i]
    index = rank.index(max(rank))
    return cluster_cnt[index]


def rankify(arr: List, is_revers=False):
    temp_arr = [i for i in arr]
    rank = [0 for i in range(len(arr))]
    temp_arr.sort()

    if is_revers:
        temp_arr.reverse()

    idx = 0
    for item in arr:
        rank[idx] = temp_arr.index(item)
        idx += 1

    return rank


def create_new_profile(data_frame):
    # Instantiating a new DF row to append later
    new_profile = pandas.DataFrame(columns=data_frame.columns)

    # Adding random values for new data
    for i in new_profile.columns[1:]:
        new_profile[i] = numpy.random.randint(0, 10, 1)

    new_profile[
        'Bios'] = "Evil communicator. Writer. Introvert. Freelance zombie lover. Professional organizer. Music junkie. Falls down a lot. Troublemaker."
    # Indexing that new profile data
    new_profile.index = [data_frame.index[-1] + 1]
    return new_profile


def get_similar_profile(cluster_df, new_profile, limit=0):
    # Assigning the split variables
    X = cluster_df.drop(["Cluster #"], 1)
    y = cluster_df['Cluster #']

    ## Vectorizing
    # Instantiating the Vectorizer
    vectorizer = CountVectorizer()

    # Fitting the vectorizer to the Bios
    x = vectorizer.fit_transform(X['Bios'])

    # Creating a new DF that contains the vectorized words
    df_wrds = pandas.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

    # Concating the words DF with the original DF
    X = pandas.concat([X, df_wrds], axis=1)

    # Dropping the Bios because it is no longer needed in place of vectorization
    X.drop(['Bios'], axis=1, inplace=True)

    scaler = MinMaxScaler()

    scaler.fit_transform(X)

    # load model
    model = load("joblib/clf_model.joblib")

    vect_new_prof = vectorizer.transform(new_profile['Bios'])

    # Quick DF of the vectorized words
    new_vect_w = pandas.DataFrame(vect_new_prof.toarray(), columns=vectorizer.get_feature_names(),
                                  index=new_profile.index)

    # Concatenating the DFs for the new profile data
    new_vect_prof = pandas.concat([new_profile, new_vect_w], 1).drop('Bios', 1)

    # Scaling the Data

    # Scaling the new profile data
    new_vect_prof = pandas.DataFrame(scaler.fit_transform(new_vect_prof), columns=new_vect_prof.columns,
                                     index=new_vect_prof.index)

    # Predicting the New Profile data by determining which Cluster it would belong to
    designated_cluster = model.predict(new_vect_prof)

    print(f"Predicting the New Profile data by determining which Cluster it would belong to:"
          f" {designated_cluster[0]}")

    des_cluster = cluster_df[cluster_df['Cluster #'] == designated_cluster[0]]

    # Appending the new profile data
    des_cluster = des_cluster.append(new_profile, sort=False)

    # Fitting the vectorizer to the Bios
    vectorizer = CountVectorizer()
    cluster_x = vectorizer.fit_transform(des_cluster['Bios'])

    # Creating a new DF that contains the vectorized words
    cluster_v = pandas.DataFrame(cluster_x.toarray(), index=des_cluster.index,
                                 columns=vectorizer.get_feature_names())

    # Joining the Vectorized DF to the previous DF and dropping columns
    des_cluster = des_cluster.join(cluster_v).drop(['Bios', 'Cluster #'], axis=1)

    # Correlations to find similar profiles
    # Finding the Top 10 similar or correlated users to the new user
    user_n = new_profile.index[0]

    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    if limit >= 0:
        top_sim = corr.sort_values(ascending=False)[1:limit]
    else:
        top_sim = corr.sort_values(ascending=False)

    return top_sim


def vectorization_1(df, columns, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    column_name = columns[0]

    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'TV', 'Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
        return df, input_df

    # Encoding columns with respective values
    if column_name in ['Religion', 'Politics']:

        # Getting labels for the original df
        df[column_name.lower()] = df[column_name].cat.codes

        # Dictionary for the codes
        d = dict(enumerate(df[column_name].cat.categories))

        d = {v: k for k, v in d.items()}

        # Getting labels for the input_df
        input_df[column_name.lower()] = d[input_df[column_name].iloc[0]]

        # Dropping the column names
        input_df = input_df.drop(column_name, 1)

        df = df.drop(column_name, 1)

        return vectorization_1(df, df.columns, input_df)

    # Vectorizing the other columns
    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()

        # Fitting the vectorizer to the columns
        x = vectorizer.fit_transform(df[column_name].values.astype('U'))

        y = vectorizer.transform(input_df[column_name].values.astype('U'))

        # Creating a new DF that contains the vectorized words
        df_wrds = pandas.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

        y_wrds = pandas.DataFrame(y.toarray(), columns=vectorizer.get_feature_names(), index=input_df.index)

        # Concating the words DF with the original DF
        new_df = pandas.concat([df, df_wrds], axis=1)

        y_df = pandas.concat([input_df, y_wrds], 1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)

        y_df = y_df.drop(column_name, 1)

        return vectorization_1(new_df, new_df.columns, y_df)


def vectorization(df, columns):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """
    column_name = columns[0]

    # Checking if the column name has been removed already
    if column_name not in ['Bios', 'TV', 'Movies', 'Religion', 'Music', 'Politics', 'Social Media', 'Sports']:
        return df

    if column_name in ['Religion', 'Politics']:
        df[column_name.lower()] = df[column_name].cat.codes

        df = df.drop(column_name, 1)

        return vectorization(df, df.columns)

    else:
        # Instantiating the Vectorizer
        vectorizer = CountVectorizer()

        # Fitting the vectorizer to the Bios
        x = vectorizer.fit_transform(df[column_name])

        # Creating a new DF that contains the vectorized words
        df_wrds = pandas.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

        # Concating the words DF with the original DF
        new_df = pandas.concat([df, df_wrds], axis=1)

        # Dropping the column because it is no longer needed in place of vectorization
        new_df = new_df.drop(column_name, axis=1)

        return vectorization(new_df, new_df.columns)


def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()

    scaler.fit(df)

    input_vect = pandas.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)

    return input_vect


def top_similar(df, cluster, vect_df, input_vect, limit=0):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster #'] == cluster[0]].drop('Cluster #', 1)

    # Appending the new profile data
    des_cluster = des_cluster.append(input_vect, sort=False)

    # Finding the Top similar or correlated users to the new user
    user_n = input_vect.index[0]

    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the Top 10 most similar profiles
    if limit > 0:
        top_sim = corr.sort_values(ascending=False)[1:(limit + 1)]
    else:
        top_sim = corr.sort_values(ascending=False)

    # The Top Profiles
    top_profile = df.loc[top_sim.index]

    # Converting the floats to ints
    top_profile[top_profile.columns[1:]] = top_profile[top_profile.columns[1:]]

    return top_profile.astype('object')


def generate_user(df):
    new_profile = pandas.DataFrame(columns=df.columns, index=[df.index[-1] + 1])
    new_profile[
        'Bios'] = "Twitteraholic. Extreme web fanatic. Food buff. Infuriatingly humble entrepreneur."
    # Adding random values for new data
    for i in new_profile.columns[1:]:
        if i in ['Religion', 'Politics']:
            new_profile[i] = numpy.random.choice(combined[i], 1, p=p[i])

        elif i == 'Age':
            new_profile[i] = halfnorm.rvs(loc=18, scale=8, size=1).astype(int)

        else:
            new_profile[i] = list(numpy.random.choice(combined[i], size=(1, 3), p=p[i]))

            new_profile[i] = new_profile[i].apply(lambda x: list(set(x.tolist())))

    return new_profile


def get_similar_profile_refined(data_frame, vect_df, new_profile):
    df = data_frame.copy()
    model = load("joblib/refined_model.joblib")
    # Applying the function to each user bio
    df['Bios'] = df.Bios.apply(tokenize)
    new_profile['Bios'] = new_profile.Bios.apply(tokenize)
    # Looping through the columns and applying the string_convert() function (for vectorization purposes)
    for col in df.columns:
        df[col] = df[col].apply(string_convert)

        new_profile[col] = new_profile[col].apply(string_convert)

    df_v, input_df = vectorization_1(df, df.columns, new_profile)

    # Scaling the New Data
    new_df = scaling(df_v, input_df)

    # Predicting/Classifying the new data
    cluster = model.predict(new_df)

    print(f"Predicting the New Profile data by determining which Cluster it would belong to:"
          f" {cluster}")

    # Finding the top 10 related profiles
    top_10_df = top_similar(data_frame, cluster, vect_df, new_df, 10)

    print(top_10_df)
