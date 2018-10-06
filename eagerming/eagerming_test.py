from utility.cross_validation import split_5_folds
from configx.configx import ConfigX
from model.trust_svd import TrustSVD

if __name__ == "__main__":
    configx = ConfigX()
    configx.k_fold_num = 5
    configx.rating_path = "../data/ft_ratings.txt"
    configx.rating_cv_path = "../data/cv/"


    # split_5_folds(configx)

    bmf = TrustSVD()
    bmf.train_model(0)
    coldrmse = bmf.predict_model_cold_users()
    print('cold start user rmse is :' + str(coldrmse))
    bmf.show_rmse()