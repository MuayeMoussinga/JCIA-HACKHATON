import streamlit as st 
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os


#pipeline avant d"etre passé au modèle
def pipeline(img):
    #Convertir PIL → OpenCV (BGR)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #Charger le modèle YOLO
    model = YOLO("best.pt")
    #Faire la prédiction
    results = model(opencv_image)
    result = results[0]
    boxes = result.boxes
    names = model.names
    #Dessiner les détections : classe, id, score
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names[cls_id]} (ID:{cls_id}) {conf:.2f}"

        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(opencv_image, label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    #Enregistrer l'image annotée
    os.makedirs("resultats", exist_ok=True)
    output_path = "resultats/image_annotee.jpg"
    cv2.imwrite(output_path, opencv_image)

    return names[cls_id]





#fonction main
def main():
    #application d'un style sur le logo agria
    st.markdown("""
    <style>
    [data-testid=stSidebar] [data-testid=stImage] {
        width: 100%; 
        margin-left: 0; 
        margin-right: 0; 
    }
    </style>
    """, unsafe_allow_html=True)
    #affichage du logo
    with st.sidebar:
        st.image("agria.png")
    
    #Titre de la page
    st.title("\U0001F680"+"Application web intelligente"+"\U0001F9E0"+" pour la classification de prunes")
    #Ajout de texte dans la side bar
    st.sidebar.write("**Hello**"+ "\U0001F60A" +" "+ "\U0001F44B")
    st.sidebar.markdown(
        "*Cette application est destinée à la classification de prunes en 6 catégories.Vous pouvez uploader une image  pour déterminer la classe de cette dernière.*\n\n"
        "**Sommaire**\n\n"
        "1. Contexte\n\n"
        "2. Modèle Yolov8\n\n"
        "3. Methodologie\n\n"
        "4. Métrique de performance\n\n"
        "5. Architecture de la solution\n\n"
        "6. Démonstration de modèle\n\n"
        
    )

    #Ajout de la couverture sur la page principale
    image = Image.open("couverture.jpg")
    # Redimensionner l'image (largeur, hauteur)
    image_resized = image.resize((700, 100))  # Largeur = 600px, Hauteur = 300px
    # Afficher l'image redimensionnée
    st.image(image_resized)
    
    #Ajout du contenu de la page
    st.subheader("1. Contexte")
    st.write("Le projet de tri automatique des prunes est lancé à l'occasion de **la journée Internationale de l'intelligence Artificielle**.\n\nIl traite le problème de l'automatisation du tri des prunes en fonction de leur état de qualité, afin de faciliter leur sélection et leur commercialisation. Actuellement, le tri des prunes repose souvent sur une inspection manuelle, ce qui peut être lent, sujet à des erreurs humaines et coûteux en main-d'œuvre. L'objectif du projet est donc de concevoir un modèle d'intelligence artificielle capable d'analyser des images de prunes et de les classer automatiquement en six catégories :\n\n")
    st.write("""
            - **Bonne qualité** : Prunes parfaitement mûres et sans défaut.
            - **Non mûre** : Prunes qui n’ont pas encore atteint leur pleine maturation.
            - **Tachetée** : Prunes présentant des taches sur leur peau.
            - **Fissurée** : Prunes ayant des fissures visibles.
            - **Meurtrie** : Prunes présentant des signes de dommage mais encore consommables.
            - **Pourrie** : Prunes en état de décomposition et impropres à la consommation.
            """)

    st.subheader("2. Modèle Yolov8")
    st.write("**Yolov8** a été notre choix pour la réalisation de ce projet car il est réputé pour :\n\n")
    st.write("""
            - **Efficacité et Rapidité** : YOLOv5 est connu pour sa rapidité dans la détection d’objets. Il traite les images en une seule passe, ce qui le rend extrêmement efficace pour des applications en temps réel.
            - **Précision Optimisée** : Grâce à ses architectures améliorées et à ses modèles pré-entraînés, YOLOv5 offre une très bonne précision de détection et de classification des objets.
            - **Facilité d’Utilisation** : YOLOv5 est construit sur PyTorch, ce qui le rend relativement simple à utiliser et à entraîner avec un jeu de données comme African Plums Dataset sur Kaggle.
             """)
    st.subheader("3. Methodologie")
    st.write("Pour réaliser cette application nous avons découpé le projet en deux parties :\n\n")
    st.write("""
            - **Front-end** : Streamlit
            - **Back-end** : Yolov8\n\n
            """
            " L'entrainement de notre modèle Yolov8 a été découpé en plusieurs étapes :\n\n")
    st.write("""
            - Nettoyage des données
            - Redimensionnement des images
            - Correction des fichiers d'images
            - Labelisation des images avec Makesense.ia
            - Entrainement du modèle sur Google Colab avec des images de Africa Plums
            - Exportation du modèle en fichier pt
             """)
    st.subheader("4. Métrique de perfomance")
    st.write("Après entrainement du modèle sur collab , voici les différnetes remarques sur son entrainement et ses perfomances : ")
    #Ajout des photos sur la page
    st.write("***1-Perfomance du modèle par classe***")
    image2 = Image.open("Model_summary/Tableau.PNG")
    st.image(image2)
    st.write("***2-Matrixe de confusion normalisée du modèle***")
    image3 = Image.open("Model_summary/confusion_matrix_normalized.png")
    st.image(image3)
    ##############################
    st.subheader("5. Architecture de la solution")
    st.write("**Architecture Générale**\n\n")
    st.write("""
             1. **Interface Utilisateur**: Conception de l'interface avec **Streamlit**, permettant aux utilisateurs d'importer des images , visualiser les résultats,chargement dynamique des données et résultats prédits par **le modèle YOLO**.
             2. **Modèle de Détection** : Intégration du modèle YOLO pour la détection d'objets entrainé sur le dataset Africa Plums.**Pipeline de traitement d'images(Redimensionnement-détection d'objets-retour des résultats au front-end)**
             """)
    st.subheader("6. Demonstration de Modèle")
    ##chargement du modèle
    #prune_classify = 
    ##################
    st.write("***Appuyer sur le bouton pour uploader l'image***")
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])
    img_title=st.text_input(label="Ajouter un titre à l'image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=img_title, use_container_width=True)
        #ajout du style au button
        st.markdown("""
                <style>
                div.stButton > button {
                    background-color: #4CAF50; /* Vert */
                    color: white;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-size: 16px;
                }
                </style>
            """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            div.stButton > button:hover {
                color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)
        #si on clique sur le button
        if (st.button("Cliquer pour déterminer la catégorie de cette prune")):
            #pipeline de traitement de l'image et prédiction
            classe=pipeline(image)
            #Recharger l'image avec Pillow et l'afficher
            final_img = cv2.imread("resultats/image_annotee.jpg")
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            final_img = Image.fromarray(final_img)
            st.write("Classification de la prune")
            st.image(final_img, use_container_width=True)
            st.markdown("**Cette prune est de la catégorie: " + classe + " 😊**")
            
            
    else:
        st.write("Erreur lors du chargement de l'image")

if __name__=='__main__':
    main()