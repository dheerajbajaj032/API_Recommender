import graphlab
import pandas as pd


class Main_Model:

    def __init__(self, base, test):
        self.base = base
        self.test = test
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_base = pd.read_csv(self.base, sep='\t', names=r_cols, encoding='latin-1')
        self.ratings_test = pd.read_csv(self.test, sep='\t', names=r_cols, encoding='latin-1')

    def create(self):
        train_data = graphlab.SFrame(self.ratings_base)
        test_data = graphlab.SFrame(self.ratings_test)
        self.item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id',
                                                                     item_id='movie_id', target='rating',
                                                                     similarity_type='cosine')
        # item_sim_recomm = item_sim_model.recommend(users=range(1, 6), k=5)
        # a = item_sim_recomm

    def predict(self, item_sim_model):
        return item_sim_model.predict(test_data)
