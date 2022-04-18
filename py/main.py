#
# Performed by Anton Kremlev
#

import argparse
import config
import pyrebase

if __name__ == '__main__':
    print("[MAIN.PY] launched")
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--person_id', help='person_id', type=str)
    parser.add_argument('-t', '--timestamp', help='timestamp', type=int)
    parser.add_argument('-p', '--prob', help='prob', type=float)
    args = parser.parse_args()

    firebase = pyrebase.initialize_app(config.firebaseConfig)
    db = firebase.database()

    prob_list = db.child(config.userId).child(config.deviceId).child("pId:" + args.person_id).child(
        "prob").get().val()
    timestamp_list = db.child(config.userId).child(config.deviceId).child("pId:" + args.person_id).child(
        "timestamp").get().val()

    # empty person
    if prob_list is None or timestamp_list is None:
        db.child(config.userId).child(config.deviceId).child("pId:" + args.person_id).update(
            {'prob': [args.prob],
             'timestamp': [args.timestamp]})
        db.child(config.userId).child(config.deviceId).update({'type': config.deviceType})
        print("[MAIN.PY] New person created: ", end='')
        print(db.child(config.userId).child(config.deviceId).get().val())
    else:
        prob_list.append(args.prob)
        timestamp_list.append(args.timestamp)
        db.child(config.userId).child(config.deviceId).child("pId:" + args.person_id).update(
            {'prob': prob_list,
             'timestamp': timestamp_list})
        print("[MAIN.PY] person " + args.person_id + " updated: ", end='')
        print(db.child(config.userId).child(config.deviceId).get().val())
