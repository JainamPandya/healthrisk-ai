import pandas as pd

from sklearn.model_selection import train_test_split

from healthrisk.preprocessing import create_preprocessor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


df = pd.read_csv(DATA_FILE, low_memory=False)

X = df.drop(columns=["readmitted", "early_readmission"])
y = df["early_readmission"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocessor = create_preprocessor(df)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Original training shape:", X_train.shape)
print("Original test shape:", X_test.shape)

print("Processed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

print("Training target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())