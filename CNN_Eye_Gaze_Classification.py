#Učitavanje potrebnih biblioteka
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


main_path = "TrainingSet/TrainingSet"
img_size = (64, 64)
batch_size = 64


#Učitavanje trening i validacionog skupa iz direktorijuma
Xtrain, Xval = image_dataset_from_directory(
    main_path,
    subset='both',
    validation_split=0.2,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=123
)


classes = Xtrain.class_names
num_classes = len(classes)
print("Dataset ima klase:", classes)
    #Found 61073 files belonging to 8 classes. 
    #Using 48859 files for training. 
    #Using 12214 files for validation. 
    #Dataset ima sledece klase: ['BottomCenter', 'BottomLeft', 'BottomRight', 'MiddleLeft', 'MiddleRight', 'TopCenter', 'TopLeft', 'TopRight']


#Prikaz podataka iz svake klase - po jedan primer iz trening skupa 
examples = {} 
for imgs, labels in Xtrain: # Prolazi kroz batch-eve slika i labela iz trening skupa
    for img, lab in zip(imgs, labels):  # Prolazi kroz svaku sliku i odgovarajuću labelu u batch-u
        lab = lab.numpy() 
        if lab not in examples: # Ako za tu klasu još nismo sačuvali primer, čuvamo sliku
            examples[lab] = img.numpy() 
        if len(examples) == len(classes):# Kada sakupimo po jedan primer iz svake klase, prekidamo unutrašnju petlju
            break 
    if len(examples) == len(classes): 
        break 
plt.figure(figsize=(10,5)) 
for i, (lab, img) in enumerate(examples.items()): 
    plt.subplot(2, 4, i+1) 
    plt.imshow(img.astype('uint8')) 
    plt.title(classes[lab]) 
    plt.axis('off') 
    plt.tight_layout() 
    plt.show()


#Histogram raspodele odbirka 
labels_all = []
for imgs, labels in Xtrain: 
    labels_all.extend(labels.numpy())
labels_all = np.array(labels_all)
plt.figure(figsize=(8,5))
plt.hist(labels_all, bins=num_classes, edgecolor='black')
plt.xticks(range(num_classes), classes, rotation=45)
plt.title('Distribucija klasa u trening skupu')
plt.xlabel('Klasa')
plt.ylabel('Broj uzoraka')
plt.grid()
plt.tight_layout()
plt.show()



#Definisanje sloja za augmentaciju podataka
data_augmentation = Sequential([
    layers.Input((img_size[0], img_size[1], 3)),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3)
])


#Prikaz augmentovanih primera
#Logika slična kao kod prikaza primera iz svake klase, ali sada se slike augmentuju pre nego što se sačuvaju kao primeri.
examples = {} 
for imgs, labels in Xtrain: 
    for img, lab in zip(imgs, labels): 
        img = data_augmentation(tf.expand_dims(img, 0))[0] 
        lab = lab.numpy() 
        if lab not in examples: 
            examples[lab] = img.numpy() 
            if len(examples) == len(classes): 
                break 
        if len(examples) == len(classes): 
            break 
plt.figure(figsize=(10,5)) 
for i, (lab, img) in enumerate(examples.items()): 
    plt.subplot(2, 4, i+1) 
    plt.imshow(img.astype('uint8')) 
    plt.title(classes[lab]) 
    plt.axis('off') 
    plt.tight_layout() 
    plt.show()

#Treniranje konvolucione mreže
def cnn_model(num_classes):
    model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

model = cnn_model(num_classes)
model.summary()


es = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    Xtrain,
    epochs=30,
    validation_data=Xval,
    callbacks=[es],
    verbose=1
)


plt.figure(figsize=(10,4))
plt.subplot(121)
# Iscrtavanje tačnosti na trening skupu i validacionom skupu kroz epohe
plt.plot(history.history['accuracy'], label="train")
plt.plot(history.history['val_accuracy'], label="val")
plt.title("Accuracy")
plt.legend()
plt.subplot(122)
# Iscrtavanje funkcije gubitka na trening skupu i validacionom skupu kroz epohe
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="val")
plt.title("Loss")
plt.legend()
plt.show()


#Predikcija i evaluacija modela na validacionom skupu
y_valtrue = []
y_valpred = []
for img, lab in Xval:
    preds = model.predict(img, verbose=0)
    y_valpred.extend(np.argmax(preds, axis=1))
    y_valtrue.extend(lab.numpy())
print("Tacnost:", accuracy_score(y_valtrue, y_valpred))
print("Precison:", precision_score(y_valtrue, y_valpred, average='weighted'))
print("Recall:", recall_score(y_valtrue, y_valpred, average='weighted'))
print("F1 score:", f1_score(y_valtrue, y_valpred, average='weighted'))


#Predikcija i evaluacija modela na trening skupu
y_traintrue = []
y_trainpred = []
for img, lab in Xtrain:
    preds = model.predict(img, verbose=0)
    y_trainpred.extend(np.argmax(preds, axis=1))
    y_traintrue.extend(lab.numpy())

#Matrica konfuzije na trening skupu
cm = confusion_matrix(y_traintrue, y_trainpred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues')
plt.title("Matrica konfuzije")
plt.show()

#Matrica konfuzije na validacionom skupu
cm = confusion_matrix(y_valtrue, y_valpred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues')
plt.title("Matrica konfuzije")
plt.show()




# Funkcija za prikaz tačno i netačno klasifikovanih slika
def show_good_bad(model, dataset, classes, title="Dobro i loše klasifikovane slike"):
    good = []  # lista tačno klasifikovanih slika
    bad = []   # lista netačno klasifikovanih slika
    # prolazak kroz dataset
    for imgs, labels in dataset:
        preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
        for i in range(len(labels)):
            true_label = labels[i].numpy()
            pred_label = preds[i]
            # čuvanje tačno klasifikovanih slika
            if pred_label == true_label and len(good) < 8:
                good.append((imgs[i].numpy(), true_label))
            # čuvanje netačno klasifikovanih slika
            elif pred_label != true_label and len(bad) < 8:
                bad.append((imgs[i].numpy(), pred_label))
        # prekid kada sakupimo dovoljno primera
        if len(good) == 8 and len(bad) == 8:
            break

    plt.figure(figsize=(12,5))
    # tačno klasifikovane slike
    for i, (im, lab) in enumerate(good):
        plt.subplot(2, 8, i+1)
        plt.imshow(im.astype("uint8"))
        plt.title(f"Tacno: {classes[lab]}")
        plt.axis("off")
    # netačno klasifikovane slike
    for i, (im, lab) in enumerate(bad):
        plt.subplot(2, 8, 8+i+1)
        plt.imshow(im.astype("uint8"))
        plt.title(f"Netacno: {classes[lab]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# poziv funkcije
show_good_bad(model, Xtrain, classes, title="Trening skup")
show_good_bad(model, Xval, classes, title="Validacioni skup")