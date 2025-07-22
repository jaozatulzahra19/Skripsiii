import joblib

def check_model_compatibility(model, vectorizer):
    # Get number of features the model expects
    if hasattr(model, 'n_features_in_'):
        model_features = model.n_features_in_
    else:
        # For older scikit-learn versions or different models
        model_features = model.coef_.shape[1]
    
    # Get number of features in vectorizer
    vectorizer_features = len(vectorizer.get_feature_names_out())
    
    if model_features == vectorizer_features:
        print(f"✅ Compatible: Model expects {model_features} features, vectorizer has {vectorizer_features}")
        return True
    else:
        print(f"❌ Incompatible: Model expects {model_features} features, vectorizer has {vectorizer_features}")
        return False

# Usage
model = joblib.load("model/svm_model.pkl")
tfidf = joblib.load("model/vectorizer.pkl")
is_compatible = check_model_compatibility(model, tfidf)