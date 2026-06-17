from hy_rec import EnhancedRecommender

if __name__=="__main__":

    rec = EnhancedRecommender(
        mode="inference"
    )

    rec.load_model()

    metrics = rec.evaluate()

    print(metrics)
