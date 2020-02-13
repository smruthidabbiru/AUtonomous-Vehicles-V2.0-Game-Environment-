"""
Data collection module (saves data in H5 format).
Saves screen captures and pressed keys into a file
for further trainings of NN.
"""

import os
import threading
import time
import winsound

import h5py

from data_collection.gamepad_cap import Gamepad
from data_collection.img_process import img_process
from data_collection.key_cap import key_check

lock = threading.Lock()

# open the data file
path = "training_data.h5"
data_file = None
if os.path.isfile(path):
    data_file = h5py.File(path, 'a')
else:
    data_file = h5py.File(path, 'w')
    # Write data in chunks for faster writing and reading by NN
    data_file.create_dataset('img', (0, 240, 320, 3), dtype='u1',
                             maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))


def save(data_img, controls, metrics):
    with lock:  # make sure that data is consistent
        if data_img:  # if the list is not empty
            # last_time = time.time()
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            data_file["metrics"].resize((data_file["metrics"].shape[0] + len(metrics)), axis=0)
            data_file["metrics"][-len(metrics):] = metrics
            # print('Saving took {} seconds'.format(time.time() - last_time))


def delete(session):
    frames = session if session < 500 else 500
    data_file["img"].resize((data_file["img"].shape[0] - frames), axis=0)
    data_file["controls"].resize((data_file["controls"].shape[0] - frames), axis=0)
    data_file["metrics"].resize((data_file["metrics"].shape[0] - frames), axis=0)


def main():
    # initialize gamepad
    gamepad = Gamepad()
    gamepad.open()

    # last_time = time.time()   # to measure the number of frames
    alert_time = time.time()  # to signal about exceeding speed limit
    close = False  # to exit execution
    pause = True  # to pause execution
    session = 0  # number of frames recorded in one session
    training_img = []  # lists for storing training data
    controls = []
    metrics = []

    print("Press RB on your gamepad to start recording")
    while not close:
        while not pause:
            # read throttle and steering values from the gamepad
            throttle, steering = gamepad.get_state()
            # get screen, speed and direction
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")

            training_img.append(screen)
            controls.append([throttle, steering])
            metrics.append([speed, direction])
            session += 1

            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            # save the data every 30 iterations
            if len(training_img) % 30 == 0:
                # print("-" * 30 + "Saving" + "-" * 30)
                threading.Thread(target=save, args=(training_img, controls, metrics)).start()
                training_img = []
                controls = []
                metrics = []

            time.sleep(0.015)  # in order to slow down fps
            # print('Main loop took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            if gamepad.get_RB():
                pause = True
                print('Paused. Save the last 15 seconds?')

                keys = key_check()
                while ('Y' not in keys) and ('N' not in keys):
                    keys = key_check()

                if 'N' in keys:
                    delete(session)
                    training_img = []
                    controls = []
                    metrics = []
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB.')
                session = 0
                time.sleep(0.5)

        if gamepad.get_RB():
            pause = False
            print('Unpaused')
            time.sleep(1)
        elif gamepad.get_LB():
            gamepad.close()
            close = True
            print('Saving data and closing the program.')
            save(training_img, controls, metrics)

    data_file.close()


if __name__ == '__main__':
    main()
