import pickle
import random

import pandas
from joblib import dump
from numpy import unique
from sklearn.utils.tests.test_pprint import CountVectorizer

from clean_analyzing_bios import bigrams, tokenize
from utils import (refining_profile_data,
                   finding_number_of_clusters_refined_data,
                   agglomerative_clustering, clustering,
                   vectorized_words_count_vector, find_list_profile_by_cluster,
                   find_best_model_classification_of_new_profile, mean_shift_clustering, scaling_categories,
                   pca_data_frame_99, finding_the_right_number_of_clusters, create_new_profile,
                   get_similar_profile, generate_user, get_similar_profile_refined)


def test_find_k_clustering():
    # Loading in the cleaned DF
    with open("pickles/profiles1.pkl", 'rb') as fp:
        data_frame = pickle.load(fp)

        # HAC-Clustering-Profiles-CountV ==> best
        clustered_df, vect_df = finding_the_right_number_of_clusters(data_frame,
                                                                     fn_vectorized_words=vectorized_words_count_vector,
                                                                     fn_algorithm_clustering=agglomerative_clustering)

        # HAC-Clustering-Profiles-CountV-Bigrams
        # finding_the_right_number_of_clusters(data_frame, fn_vectorized_words=vectorized_words_count_vector,
        #                                      fn_algorithm_clustering=agglomerative_clustering,
        #                                      is_bigrams=True)

        # HAC-Clustering-Profiles-TFIDF
        # finding_the_right_number_of_clusters(data_frame, fn_vectorized_words=vectorized_words_tfidf,
        #                                      fn_algorithm_clustering=agglomerative_clustering)

        # HAC-Clustering-Profiles-TFIDF-Bigrams
        # finding_the_right_number_of_clusters(data_frame, fn_vectorized_words=vectorized_words_tfidf,
        #                                      fn_algorithm_clustering=agglomerative_clustering,
        #                                      is_bigrams=True)
        # Kmeans-Clustering-Profiles-CountV
        # finding_the_right_number_of_clusters(data_frame, fn_vectorized_words=vectorized_words_count_vector,
        #                                      fn_algorithm_clustering=k_means_clustering)

        # Kmeans-Clustering-Profiles-TFIDF
        # finding_the_right_number_of_clusters(data_frame, fn_vectorized_words=vectorized_words_tfidf,
        #                                      fn_algorithm_clustering=k_means_clustering)
        with open("pickles/clustered_profiles.pkl", "wb") as wb:
            clustered_df.to_csv(r"csv/clustered_profiles.csv", index=False)
            pickle.dump(clustered_df, wb)
        with open("pickles/vectorized_profiles.pkl", "wb") as wb:
            vect_df.to_csv(r"csv/vectorized_profiles.csv", index=False)
            pickle.dump(vect_df, wb)


def test_clustering(n_clusters):
    # Loading in the cleaned DF
    with open("pickles/profiles1.pkl", 'rb') as fp:
        data_frame = pickle.load(fp)
        clustered_df = clustering(data_frame, fn_vectorized_words=vectorized_words_count_vector,
                                  fn_algorithm_clustering=agglomerative_clustering, n_clusters=n_clusters)
        with open("pickles/clustered_profiles.pkl", "wb") as wb:
            clustered_df.to_csv(r"csv/clustered_profiles.csv", index=False)
            pickle.dump(clustered_df, wb)


def test_bigrams():
    with open("pickles/profiles1.pkl", 'rb') as fp:
        data_frame = pickle.load(fp)
        bigram_df = bigrams(data_frame)
        with open("pickles/clean_bigram_df.pkl", "wb") as wp:
            bigram_df.to_csv(r"csv/clean_bigram_df.csv")
            pickle.dump(bigram_df, wp)


def test_refining_profile_data():
    with open("pickles/profiles.pkl", 'rb') as rb:
        df = pickle.load(rb)
        df.to_csv(r'csv/profiles.csv', index=False)
        df_new = refining_profile_data(df)
        with open("pickles/refined_profiles.pkl", 'wb') as wb:
            df_new.to_csv(r'csv/refined_profiles.csv')
            pickle.dump(df_new, wb)
            print("done!!!")


def test_finding_number_of_clusters_refined_data():
    # Loading in the cleaned DF
    with open("pickles/refined_profiles.pkl", 'rb') as fp:
        df = pickle.load(fp)
        cluster_df, vect_df = finding_number_of_clusters_refined_data(df,
                                                                      fn_algorithm_clustering=agglomerative_clustering)

        with open("pickles/refined_cluster.pkl", 'wb') as wb:
            cluster_df.to_csv(r'csv/refined_cluster.csv')
            pickle.dump(cluster_df, wb)

        with open("pickles/vectorized_refined.pkl", 'wb') as wb:
            vect_df.to_csv(r'csv/vectorized_refined.csv')
            pickle.dump(vect_df, wb)


def test_clustering_refined_data(n_clusters):
    with open("pickles/refined_profiles.pkl", 'rb') as fp:
        df = pickle.load(fp)
        cluster_df, vect_df = finding_number_of_clusters_refined_data(df,
                                                                      fn_algorithm_clustering=agglomerative_clustering)
        with open("pickles/refined_cluster.pkl", 'wb') as wb:
            cluster_df.to_csv(r'csv/refined_cluster.csv')
            pickle.dump(cluster_df, wb)

        with open("pickles/vectorized_refined.pkl", 'wb') as wb:
            vect_df.to_csv(r'csv/vectorized_refined.csv')
            pickle.dump(vect_df, wb)


def test_find_list_profile_by_cluster():
    with open("pickles/clustered_profiles.pkl", "rb") as fp:
        df = pickle.load(fp)
        corr_group = find_list_profile_by_cluster(df, CountVectorizer, 3)
        corr_group.to_csv(r"csv/corr_group.csv")
        random_user = random.choice(corr_group.index)
        print("Top 10 most similar users to User #", random_user, '\n')

        top_10_sim = corr_group[[random_user]].sort_values(by=[random_user], axis=0, ascending=False)[1:11]
        top_10_sim.to_csv(r"csv/top_10_sim.csv")
        print(top_10_sim)

        print("\nThe most similar user to User #", random_user, "is User #", top_10_sim.index[0])


def test_find_best_model_classification_of_new_profile():
    with open("pickles/vectorized_refined.pkl", 'rb') as fp:
        vect_df = pickle.load(fp)
        best_model = find_best_model_classification_of_new_profile(vect_df)
        # Saving the Classification Model For future use
        dump(best_model, "joblib/refined_model.joblib")

    # with open("pickles/vectorized_profiles.pkl", 'rb') as fp:
    #     vect_df = pickle.load(fp)
    #     best_model = find_best_model_classification_of_new_profile(vect_df)
    #     # Saving the Classification Model For future use
    #     dump(best_model, "joblib/clf_model.joblib")


def test_mean_shift():
    with open("pickles/profiles1.pkl", 'rb') as fp:
        data_frame = pickle.load(fp)
        # Applying the function to each user bio
        data_frame['Bios'] = data_frame.Bios.apply(tokenize)

        scale_df = scaling_categories(data_frame)

        # Creating a new DF that contains the vectorized words
        df_words = vectorized_words_count_vector(scale_df['Bios'])

        # Concating the words DF with the original DF
        new_df = pandas.concat([scale_df, df_words], axis=1)

        # Dropping the Bios because it is no longer needed in place of vectorization
        new_df.drop('Bios', axis=1, inplace=True)

        # Fitting and transforming the dataset to the stated number of features
        df_pca = pca_data_frame_99(new_df)

        cluster_assignments = mean_shift_clustering(df_pca)

        data_frame["Cluster #"] = cluster_assignments

        print(unique(cluster_assignments))

        with open("pickles/mean_shift_clustering.pkl", "wb") as wb:
            data_frame.to_csv(r"csv/mean_shift_clustering.csv", index=False)
            pickle.dump(data_frame, wb)


def test_get_similar_profile():
    with open("pickles/clustered_profiles.pkl", 'rb') as fp_clustered:
        clustered_profiles = pickle.load(fp_clustered)
        with open("pickles/profiles1.pkl", 'rb') as fp_profile:
            data_frame = pickle.load(fp_profile)
            new_profile = create_new_profile(data_frame)
            top_sim = get_similar_profile(clustered_profiles, new_profile, 100)
            data_frame.append(new_profile)
            df = data_frame.loc[top_sim.index]
            df


def test_get_similar_profile_refined():
    # Loading the Profiles
    with open("pickles/refined_profiles.pkl", 'rb') as dffp:
        df = pickle.load(dffp)

    with open("pickles/vectorized_refined.pkl", 'rb') as vect_dffp:
        vect_df = pickle.load(vect_dffp)
    new_profile = generate_user(df)
    new_profile.to_csv(r"csv/new_profile.csv")
    print(new_profile.head())
    get_similar_profile_refined(df, vect_df, new_profile)


if __name__ == "__main__":
    # test_find_k_clustering()

    # test_bigrams()

    # test_clustering_refined_data(2)
    # test_find_list_profile_by_cluster()

    # test_mean_shift()
    # test_refining_profile_data()
    # test_finding_number_of_clusters_refined_data()
    # test_find_best_model_classification_of_new_profile()
    test_get_similar_profile_refined()
