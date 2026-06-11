from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

text = input("Enter text: ")

result = classifier(text)

print(result)