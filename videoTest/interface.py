import cv2
import mediapipe as mp
from videoTest import board, util
from videoTest import grabDetection as gD
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class interface:
    def __init__(self):
        self.no_hands = True

        # Board for Minecraft conversion
        self.board = None

        # Initialize hand detection
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

        # For webcam input:
        # cap = cv2.VideoCapture("./testVideos/IMG_4362.MOV")
        # cap = cv2.VideoCapture("./testVideos/test3.MOV")
        self.cap = cv2.VideoCapture(1)

        # List of hands
        self.loh = []

        # Log of all the actions taken
        self.log = []

        self.image = None
        self.go = True

        self.overlapping_lm = 21 // 5

        self.recording = cv2.VideoWriter('test3_using_webcam.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (640, 480))

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self.get_frames)

    def get_frames(self):
        while True:
            success, image = self.cap.read()
            if success:
                self.image = image
            else:
                self.go = False
                break

    # Get the measurement of the blocks
    def prompt_measurement(self):
        if self.board is not None:
            return

        while self.image is None:
            continue
        img = deepcopy(self.image)
        txt = 'Put Block Down For Measurement. Press "a" when complete.'
        img_txt = cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        # start_time = time.time()
        cv2.imshow('MediaPipe Hands', img_txt)
        if cv2.waitKey(0) == ord('a'):
            pass

        # ds = get_half_dimensions(img)
        # # resize image
        # img = cv2.resize(img, ds)
        img = deepcopy(self.image)
        board_ret = board.Board(img)

        txt = 'Remove block. Press "a" when complete.'
        img_txt = cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', img_txt)
        if cv2.waitKey(0) == ord('a'):
            pass

        self.board = board_ret

    def show_starting_board(self):
        img = deepcopy(self.image)
        txt = 'Tips:'
        img = cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        txt = '1. Have the camera three feet from the workspace.'
        img = cv2.putText(img, txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        txt = '2. Only hold one block at a time with your index finger and thumb.'
        img = cv2.putText(img, txt, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        txt = '3. Try to have your fingers visible to the camera'
        img_txt = cv2.putText(img, txt, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', img_txt)
        cv2.waitKey(3000)

    def overlap(self, hand1, hand2):
        count = 0
        for i in range(0, len(hand1.hl)):
            lm1 = hand1.hl[i]
            lm2 = hand2.hl[i]
            if hand1.distance_params[0] >= util.eud_dist(lm1[0], lm1[1], lm2[0], lm2[1]):
                count += 1
        print(count)
        return count > self.overlapping_lm

    def process_image(self):
        while self.go:
            # Resize image
            # dsize = get_half_dimensions(image)
            dsize = (640, 480)
            image = cv2.resize(self.image, dsize)

            # TODO: Uncomment later
            # if first:
            #   show_starting_board(image)
            #   first = False

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            # For debugging purposes
            # print(frame)
            # if frame == 125:
            #     stop = 0

            # Find the hands in the images
            results = self.hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # List of coordinates where blocks have been grabbed or dropped
            lod = []

            # Get a copy of the hand info
            cmh = deepcopy(results.multi_handedness)

            # If hand info is none then there are no hands in the frame
            if cmh is not None:
                # Convert all hand point normalized coordinates to image coordinates.
                cmhl = util.hlist_to_coords(results.multi_hand_landmarks, dsize)
                ind = 0

                # Iterate through the hands
                while len(self.loh) > ind:
                    # Get current hand
                    temp_hand = self.loh[ind]

                    # See if any grab or drop occurred
                    x, y, release = temp_hand.print_toggle(cmh, cmhl, image)

                    # If a grab or drop occurred, update the board
                    if release is not None:
                        if release:
                            self.board.remove_single(x, y)
                        else:
                            self.board.add_single(x, y)

                        # Log used to keep track of what was dropped and picked up
                        self.log.append([x, y, release])

                    # If there was a grab or drop, add it to the list of coordinates to mark the location
                    if x is not None:
                        lod.append((x, y))

                    image = temp_hand.update_interface(cmh, cmhl, image, self.cap)

                    # Update all the information about the hand from the last time to this frame
                    rem = temp_hand.update_everything(cmh, cmhl)

                    # If hand is successfully tracked and updated, remove from hand results for the frame
                    # Otherwise remove from list of hands we are tracking
                    if rem is not None:
                        cmh.pop(rem)
                        cmhl.pop(rem)
                        ind += 1
                    else:
                        self.loh.pop(ind)

                # Add whatever remaining hands that didn't match any of the tracked hands to the tracked hands list
                # In other words, we begin tracking the "new" hands (might be mistakenly considered new)
                for index in range(0, len(cmh)):
                    temp_hand = gD.hand(cmhl[index], cmh[index].classification._values[0].label,
                                     cmh[index].classification._values[0].score, self.board)
                    self.loh.append(temp_hand)
                no_hands = 0

                for i in range(0, len(self.loh)):
                    for j in range(i + 1, len(self.loh)):
                        if self.overlap(self.loh[i], self.loh[j]):
                            self.loh[i].update_spazz(True, image)
                            self.loh[j].update_spazz(True, image)
            else:
                # No hands are detected so remove all the currently tracked hands and update the no_hands state
                # No hands state only begins keeping track after hands initially appear
                self.loh = []
                if self.no_hands is not None:
                    self.no_hands += 1

            # Draw the points on the hand
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # For debugging purposes
            # for h in range(0, len(loh)):
            #     print(str(h) + ": " + str(loh[h]))
            #     continue

            # if len(loh) > 0:
            #     # Test if it's moving
            #     for val in loh:
            #         print(val)
            #     print("     ")
            #     # if val.moving:
            #     #     print("moving")

            # Draw circles to mark where blocks were placed in this frame
            for d in lod:
                x = int(d[0])
                y = int(d[1])
                image = cv2.circle(image, (x, y), 50, (255, 0, 0), 10)

            # Draw grid lines to the image
            image = util.drawlines(dsize, image, self.board)

            # Draw marks on where blocks are located at all times
            for i in range(0, len(self.board.top)):
                for j in range(0, len(self.board.top[0])):
                    if self.board.top[i][j] != 0:
                        cx, cy = self.board.centers[i][j]
                        image = cv2.circle(image, (cx, cy), 30, (0, 255, 0), 10)

            self.recording.write(image)

            # Display the new augmented frame.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.recording.close()


def main():
    display = interface()
    display.prompt_measurement()
    display.process_image()


if __name__ == "__main__":
    main()



