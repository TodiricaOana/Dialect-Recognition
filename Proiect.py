from sklearn import svm
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

with open("E:/HARD HDD/doc faculatte/IA/Proiect/train_samples.txt", encoding = "utf-8") as f:
    training_data1 = f.read()
with open("E:/HARD HDD/doc faculatte/IA/Proiect/train_labels.txt", encoding = "utf-8") as g:
    training_id_labels1 = g.read()
with open("E:/HARD HDD/doc faculatte/IA/Proiect/validation_samples.txt", encoding = "utf-8") as k:
    validation_id_data = k.read()
with open("E:/HARD HDD/doc faculatte/IA/Proiect/validation_labels.txt", encoding = "utf-8") as p:
    validation_id_labels = p.read()

training_data1 += validation_id_data # unim train_samples si validation_samples pentru a antrena pe ambele

training_id_labels1 += validation_id_labels #unim si label-urile

#scriem noile date intr un alt fisier:
with open ("E:/HARD HDD/doc faculatte/IA/Proiect/train_samples_all.txt", 'w', encoding = "utf-8") as fp:
    fp.write(training_data1)

with open ("E:/HARD HDD/doc faculatte/IA/Proiect/train_labels_all.txt", 'w', encoding = "utf-8") as fp:
    fp.write(training_id_labels1)

f1 = open("E:/HARD HDD/doc faculatte/IA/Proiect/train_samples_all.txt", "r", encoding = "utf-8")
g1 = open("E:/HARD HDD/doc faculatte/IA/Proiect/train_labels_all.txt", "r", encoding = "utf-8")
h = open("E:/HARD HDD/doc faculatte/IA/Proiect/test_samples.txt", "r", encoding = "utf-8")

#citim noile datele unite ( train + validation), atat samples, cat si labels
training_data = f1.readlines()
training_id_labels = g1.readlines()
test_data = h.readlines() #citim si datele de test


#construim o functie care elimina id-ul de la inceputul fiecarei linii, pentru a nu-l folosi la antrenare si
#care tine minte id ul fiecarei linii in parte, pentru a-l putea folosi mai tarziu la afisare
def get_features(text):
    TextWithoutId = []
    IdRow = {}

    for row in text:
        TextWithoutId.append(row.split(maxsplit=1)[1])

    for index_row, row in enumerate(text):
        for index_word, word in enumerate(row.split()):
            if index_word == 0:
                IdRow[index_row] = word
            else:
                break
    return TextWithoutId, IdRow #returneaza continutul liniilor fara id-ul de la inceput si id-ul corespunzator fiecarei linii


#construim o functie care returneaza doar labelurile din fisierele ce contin labels (elimina id-ul)
def get_labels(text):
    labels = []
    for row in text:
        for index_word, word in enumerate(row.split()):
            if index_word == 1:
                labels.append(word)
    return labels


training_data_no_Id, training_Id_Row = get_features(training_data)
test_data_no_Id, test_Id_Row = get_features(test_data)
training_labels = get_labels(training_id_labels)

vect = TfidfVectorizer(analyzer='char', ngram_range=(5,5), max_features=400000)
#construieste o matrice din datele de antrenare, cu frecventa cuvintelor normalizata si retinuta in ordine descrescatoare:
training_vectorizer = vect.fit_transform(training_data_no_Id)
#normalizeaza datele de test
test_vectorizer = vect.transform(test_data_no_Id)


svm_model = svm.LinearSVC(C=0.5, tol=1e-5)
svm_model.fit(training_vectorizer, training_labels)
predicted_labels = svm_model.predict(test_vectorizer)

#scriem in fisierul csv:
with open('sample_submission.csv', 'w', newline='') as result:
    w = csv.writer(result)
    w.writerow(["id", "label"])
    for index_row, prediction in enumerate(predicted_labels):
        w.writerow([test_Id_Row[index_row], prediction])


#Calculez f1 score ( antrenez doar pe train)

f2 = open("E:/HARD HDD/doc faculatte/IA/Proiect/train_samples.txt", "r", encoding = "utf-8")
g2 = open("E:/HARD HDD/doc faculatte/IA/Proiect/train_labels.txt", "r", encoding = "utf-8")

f3 = open("E:/HARD HDD/doc faculatte/IA/Proiect/validation_samples.txt", "r", encoding = "utf-8")
g3 = open("E:/HARD HDD/doc faculatte/IA/Proiect/validation_labels.txt", "r", encoding = "utf-8")

training_data2 = f2.readlines()
training_id_labels2 = g2.readlines()
validation_samples = f3.readlines()
validation_id_labels1 = g3.readlines()

training, training_id = get_features(training_data2)
validation, validation_id = get_features(validation_samples)

vect = TfidfVectorizer(analyzer='char', ngram_range=(5, 5), max_features=400000)
training_vectorizer1 = vect.fit_transform(training)
training_labels1 = get_labels(training_id_labels2)
validation_vectorizer = vect.transform(validation)

svm_model = svm.LinearSVC(C=1, random_state=0, tol=1e-5)
svm_model.fit(training_vectorizer1, training_labels1)
predicted_validation = svm_model.predict(validation_vectorizer)

from sklearn.metrics import f1_score
import numpy as np


#avem nevoie de un vector pentru apelarea functie f1_score:
validation_labels = np.zeros((len(validation_id_labels1)))

for index_row, row in enumerate(validation_id_labels1):
    for index_word, word in enumerate(row.split()):
        if index_word == 1:
            validation_labels[index_row] = word


#avem nevoie de un vector de numere, nu de caractere, asa cum a fost prezis pentru apelarea functiei f1_score:
predicted_labels_validation = np.zeros((len(predicted_validation)))

for index, word in enumerate(predicted_validation):
    predicted_labels_validation[index] = word


#afisam f1 score
print('f1 score:', f1_score(np.asarray(validation_labels), predicted_labels_validation))


from sklearn.metrics import confusion_matrix

#calculam matricea de confuzie
matrix = confusion_matrix(validation_labels, predicted_labels_validation)

print ('Confusion Matrix :\n', matrix)


