from fake_news_detector import fake_news_detector
try:
    fake_news_detector.load_models('models/fake_news/')
    print("Transformer loaded:", fake_news_detector.use_transformer)
except Exception as e:
    print("Exception during load_models:", e)