import lightgbm as lgb
params_lgb = {
    'objective': 'binary',
    'max_depth': 3,
    'learning_rate': 0.1,
    'metric': 'binary_logloss'
}
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
num_rounds = 100
model_lgb = lgb.train(params_lgb, train_data, num_rounds, valid_sets=[test_data])
predictions_lgb = model_lgb.predict(X_test)
binary_predictions_lgb = [1 if x > 0.5 else 0 for x in predictions_lgb]
accuracy_lgb = accuracy_score(y_test, binary_predictions_lgb)
print(f'Accuracy of LightGBM: {accuracy_lgb}')# xgbooost
