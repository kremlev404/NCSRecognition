import pyrebase

firebaseConfig = {
    'apiKey': "AIzaSyDNjvqZU1sgyycaNAUKsuNKNERqYqTohMo",
    'authDomain': "ncsrecognition.firebaseapp.com",
    'databaseURL': "https://ncsrecognition-default-rtdb.europe-west1.firebasedatabase.app",
    'projectId': "ncsrecognition",
    'storageBucket': "ncsrecognition.appspot.com",
    'messagingSenderId': "311133578336",
    'appId': "1:311133578336:web:24c82cf92cc51e918a4f05",
    'measurementId': "G-4VZCTGE7M4"
}
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    firebase = pyrebase.initialize_app(firebaseConfig)
    db = firebase.database()
    db.child("mId:kremlev404").child("rsId:raspId1").update({'type': "x86"})
    db.child("mId:kremlev404").child("rsId:raspId1").child("pId:recognizedPersonId1").update(
        {'prob': [0.70, 0.22, 0.77, 0.65],
         'timestamp': ["14.04.2022;23.00", "14.04.2022;23.05", "14.04.2022;23.10", "14.04.2022;23.15"]})
    db.child("mId:kremlev404").child("rsId:raspId2").update({'type': "raspberry"})
    db.child("mId:kremlev404").child("rsId:raspId2").child("pId:recognizedPersonId2").update(
        {'prob': [0.70, 0.22, 0.77, 0.65],
         'timestamp': ["14.04.2022;23.00", "14.04.2022;23.05", "14.04.2022;23.10", "14.04.2022;23.15"]})
    db.child("mId:kremlev404").child("rsId:raspId2").child("pId:recognizedPersonId3").update(
        {'prob': [0.20, 0.32, 0.47, 0.55],
         'timestamp': ["14.04.2022;23.00", "14.04.2022;23.05", "14.04.2022;23.10", "14.04.2022;23.15"]})
