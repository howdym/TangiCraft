import cv2
import mediapipe as mp
import math
from videoTest import util
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#########################
# ADJUSTABLE PARAMETERS #
#########################

# self.grace_period_timer
# self.grace_period_timer_threshold
# self.grace_period_reset_toggle
# self.stability_timer
# self.stability_timer_threshold
# self.distance_params
# self.grabbing_params
# function find_loc – percentage threshold
# variable trigger – line 461


class hand:
    def __init__(self, mhl_val, handedness, percentage, curr_board, grabbing=False):
        ################
        # General Info #
        ################
        # Whether it's the left or right hand
        self.handedness = handedness

        # How certain one is about whether it's the left or right hand
        self.percentage = percentage

        #################
        # Hand tracking #
        #################
        # Last location of the hand (based on wrist)
        self.last_loc = mhl_val[0]

        # State of whether the hand is moving
        self.moving = False

        # Slope of hand
        self.slope = self.get_hand_slope(mhl_val)

        ###################
        # Actual grabbing #
        ###################
        # State of whether the hand is grabbing something
        self.grabbing = grabbing

        # Keep track of the time since a hand appeared
        # The code won't start detecting if the hand is grabbing anything until the grace period is past the
        # maximum grace period set (self.max_gp)
        self.grace_period_timer = 0

        # Only bc my hand did that weird thing was 15 added
        # self.max_gp = 15
        self.grace_period_timer_threshold = 1

        # Similar role to the grace period, but different functionality
        self.stability_timer = 0

        # The threshold where we can conclude that the hand is stablized
        self.stability_timer_threshold = 30

        # Workaround for messed up hands
        # Basically saying that if the confidence of the handedness is less that 0.7, change it to None because we don't
        # actually really know what the best hand is
        if percentage >= 0.7:
            self.best_hand = handedness
        else:
            self.best_hand = None

        # The lower and upper bound to what distance a hand travels still qualifies it being the same hand we were
        # already tracking
        # Arbitrary values, need to mess around with
        self.distance_params = [curr_board.side_length / 4, curr_board.side_length / 2]

        # The lower and upper bound to what distance the fingers need to be at to qualify for the fingers to be close
        # enough (grabbing something)
        # Trying to do sqrt(2) more than side length in the case of holding it by the diagonal
        self.grabbing_params = [0, curr_board.side_length * 2]


        ###########################
        # New Interface Variables #
        ###########################
        # Spazzing: Two hands on top of each other or finger points that are moving more than just the normal threshold
        self.spazzing = False
        self.spazzing_threshold = 300
        self.spazzing_timer = 0
        self.spazzing_ratio = 0.4

        # TODO: Might merge focused with stability
        self.focused_threshold = 60
        self.focused_timer = 0

        self.hl = mhl_val

    # String representation of a hand, for debugging purposes
    def __str__(self):
        return str(self.last_loc[0]) + " " + str(self.last_loc[1]) + " " + str(self.handedness) \
               + " " + str(self.grabbing) + " " + str(self.slope)

    # Debug representation of a hand, for debugging purposes
    def __repr__(self):
        return str(self.last_loc[0]) + " " + str(self.last_loc[1]) + " " + str(self.handedness) \
               + " " + str(self.grabbing) + " " + str(self.slope)

    # Set the slope of the hand
    def set_hand_slope(self, mhl_val):
        self.slope = self.get_hand_slope(mhl_val)

    # Get the slope of the hand using the middle fingertip and wrist
    def get_hand_slope(self, mhl_val):
        wrist = mhl_val[0]
        middle_tip = mhl_val[12]
        return math.tanh((wrist[1] - middle_tip[1]) / (wrist[0] - middle_tip[0]))

    # Check if the hand is not moving
    def is_still(self, loc):
        # Get the distance from the current location and the last location
        distance = util.eud_dist(loc[0], loc[1], self.last_loc[0], self.last_loc[1])
        params = self.distance_params
        # The distance should be less than the moving lower bounds to qualify as not moving
        return distance < params[0]

    # Check if the hand is moving
    def is_moving(self, loc):
        # Get the distance from the current location and the last location
        distance = util.eud_dist(loc[0], loc[1], self.last_loc[0], self.last_loc[1])
        params = self.distance_params
        # The distance should be within the moving bounds to qualify as moving
        return params[0] < distance < params[1]

    # Check if the hand rotated by seeing whether the slope of the hand changed a certain amount
    def is_rotated(self, mhl_val):
        return abs(self.slope - self.get_hand_slope(mhl_val)) < 0.25

    # Return new location and index if there, else return None
    def find_loc(self, mh, mhl):
        for i in range(0, len(mh)):
            # Get the information about the current hand
            handedness = mh[i].classification._values[0].label
            percentage = mh[i].classification._values[0].score

            # Get the coordinates of the points of the hand
            mhl_val = mhl[i]

            # Get the wrist coordinates
            loc = mhl_val[0]

            # Check if the hand is moving. If not, then no need to update the information
            # Anything beyond these constraints is considered a different hand
            if (self.is_moving(loc) or self.is_still(loc)) and self.is_rotated(mhl_val):
                # If the handedness confidence is greater than 0.7, change the best hand value
                if percentage > 0.7:
                    self.best_hand = handedness
                # Update everything else and return the location of the hand and the index
                self.handedness = handedness
                self.percentage = percentage
                self.set_hand_slope(mhl_val)
                return loc, i
        # Otherwise, if hand is not found, return None
        return None, None

    # Update the last location of the hand
    # Return True if successful, else False
    def update_loc(self, mh, mhl):
        # Get the location of the hand
        loc, ind = self.find_loc(mh, mhl)

        # Update it if the location was found
        if loc is not None:
            self.last_loc = loc
            return True
        return False

    # Check to see if the hand is grabbing something
    # Return True if grabbing, else False
    def is_grabbing(self, mh, mhl):
        # If it's moving, just return whatever the old value was. We are assuming we can't throw items and we can't
        # pick items up that quickly
        if self.moving:
            return self.grabbing

        # Get location and index of the hand
        loc, ind = self.find_loc(mh, mhl)

        # If hand is not found, then it's not moving
        if loc is None:
            return False

        # Get the coordinates of the points on the hand
        hand_landmarks = mhl[ind]

        p = self.grabbing_params

        p2 = util.is_thumb_near_finger(hand_landmarks, 2, p[0], p[1])

        # Focus on only index finger for now
        pincher = p2
        # grab = not thumbIsOpen and pincher
        grab = pincher

        return grab

    # If the grab state has changed, update it and reset the timer for stability delay
    def update_grabbing(self, mh, mhl):
        new_grab = self.is_grabbing(mh, mhl)
        if self.grabbing != new_grab:
            self.stability_timer = 0
        self.grabbing = new_grab

    # Check if toggled from grabbing to not, and vice versa. Find loc, convert it to pic coordinates, and print it,
    # with the change. Return coordinates.
    def print_toggle(self, mh, mhl, img):
        # TODO: check if getting rid of grace_period still works
        # If grace period timer less than threshold, still in grace period, so don't check
        # if self.grace_period_timer <= self.grace_period_timer_threshold:
        #     return None, None, None

        # Get location and index of hand
        loc, ind = self.find_loc(mh, mhl)
        # If hand is not found, don't return coordinates
        if loc is not None:
            # Get location of potential drop
            thumb = mhl[ind][4]
            index = mhl[ind][8]
            mid = util.find_midpoint(thumb[0], thumb[1], index[0], index[1])
            x = mid[0]
            y = mid[1]

            # Check if the hand is stable and has toggled grab state
            grabbing = self.is_grabbing(mh, mhl)
            # TODO: Check if getting rid of stability_timer still works. Might not want to do this.
            if grabbing and not self.grabbing and not self.spazzing: # and self.stability_timer >= self.stability_timer_threshold:
                print("Grabbed at ({}, {})".format(x, y))
                return x, y, True
            if self.grabbing and not grabbing and not self.spazzing: # and self.stability_timer >= self.stability_timer_threshold:
                print("Released at ({}, {})".format(x, y))
                return x, y, False
        return None, None, None

    # Return index. Check if there first, then update grabbing, then update loc.
    # It'll be on the main function to take index and remove available entry or remove hand.
    def update_everything(self, mh, mhl):
        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            return None

        self.moving = self.is_moving(loc)
        self.update_grabbing(mh, mhl)
        self.update_loc(mh, mhl)

        # Update the times as well
        # self.grace_period_timer += 1
        # self.stability_timer += 1
        return ind

    ########################
    # Interface Algorithms #
    ########################

    # Check if the hand is moving
    def is_spazzing(self, loc, ind):
        # Get the distance from the current location and the last location
        distance = util.eud_dist(loc[0], loc[1], self.hl[ind][0], self.hl[ind][1])
        params = self.distance_params
        # The distance should be within the moving bounds to qualify as moving
        return distance >= params[1]

    def check_spazzing(self, hl):
        count = 0
        for i in range(0, len(self.hl)):
            if self.is_spazzing(hl[i], i):
                count += 1
        return count / 21 > self.spazzing_ratio

    # hl: current hand landmarks
    # img: the image of the current frame
    def update_interface(self, mh, mhl, img, cap):
        #print(img)
        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            # print("here")
            return img
        # print("33333")

        hl = mhl[ind]
        res = self.check_spazzing(hl)
        self.hl = hl
        return self.update_spazz(res, img)

    def update_spazz(self, res, img):
        new_img = img
        if res:
            self.spazzing_timer += 1
            self.focused_timer = 0
            self.spazzing = res
            txt = 'Not Focused'
            new_img = cv2.putText(new_img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2,
                                  cv2.LINE_AA)
        else:
            self.focused_timer += 1
            self.spazzing_timer = 0
        if self.spazzing_timer > self.spazzing_threshold:
            # self.show_starting_board(cap, img)
            # self.spazzing_timer = 0
            pass
        elif self.focused_timer > self.focused_threshold:
            self.spazzing = False
        elif self.focused_timer < self.focused_threshold:
            txt = 'Not Focused'
            new_img = cv2.putText(new_img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2,
                                  cv2.LINE_AA)
        return new_img


# def main():
#     # For debugging purposes
#     frame = 0
#
#     # Board for Minecraft conversion
#     board = None
#
#     # Initialize hand detection
#     hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
#
#     # For webcam input:
#     # cap = cv2.VideoCapture("./testVideos/IMG_4362.MOV")
#     # cap = cv2.VideoCapture("./testVideos/test3.MOV")
#     cap = cv2.VideoCapture(1)
#
#     # List of hands
#     loh = []
#
#     # Whether a frame has no hands in it
#     no_hands = None
#
#     # Trigger board cleanup when there are no hands for 10 consecutive frames
#     trigger = 10
#
#     # Log of all the actions taken
#     log = []
#
#     result = cv2.VideoWriter('test1_using_webcam.avi',
#                              cv2.VideoWriter_fourcc(*'MJPG'),
#                              10, (540, 960))
#
#     # Check if it's the first frame
#     first = True
#     while cap.isOpened():
#         # Read the image
#         success, image = cap.read()
#
#         if not success:
#             print("Ignoring empty camera frame.")
#             for line in board.top:
#                 print(*line)
#             pprint.pprint(board.send_info)
#             # If loading a video, use 'break' instead of 'continue'.
#             break
#
#         # Resize image
#         # dsize = get_half_dimensions(image)
#         dsize = (640, 480)
#         image = cv2.resize(image, dsize)
#
#         # Set up board if not set up already
#         if board is None:
#             # Replace with board prompt
#             # board = contourUtil.Board(cv2.resize(image, dsize))
#             board = prompt_measurement(cap, image)
#
#         # TODO: Uncomment later
#         # if first:
#         #   show_starting_board(image)
#         #   first = False
#
#         # Flip the image horizontally for a later selfie-view display, and convert
#         # the BGR image to RGB.
#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         # To improve performance, optionally mark the image as not writeable to
#         # pass by reference.
#         image.flags.writeable = False
#
#         # For debugging purposes
#         # print(frame)
#         # if frame == 125:
#         #     stop = 0
#
#         # Find the hands in the images
#         results = hands.process(image)
#
#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # List of coordinates where blocks have been grabbed or dropped
#         lod = []
#
#         # Get a copy of the hand info
#         cmh = deepcopy(results.multi_handedness)
#
#         # If hand info is none then there are no hands in the frame
#         if cmh is not None:
#             # Convert all hand point normalized coordinates to image coordinates.
#             cmhl = hlist_to_coords(results.multi_hand_landmarks, dsize)
#             ind = 0
#
#             # Iterate through the hands
#             while len(loh) > ind:
#                 # Get current hand
#                 temp_hand = loh[ind]
#
#                 # See if any grab or drop occurred
#                 x, y, release = temp_hand.print_toggle(cmh, cmhl, image)
#
#                 # If a grab or drop occurred, update the board
#                 if release is not None:
#                     if release:
#                         board.remove_single(x, y)
#                     else:
#                         board.add_single(x, y)
#
#                     # Log used to keep track of what was dropped and picked up
#                     log.append([x, y, release])
#
#                 # If there was a grab or drop, add it to the list of coordinates to mark the location
#                 if x is not None:
#                     lod.append((x, y))
#
#                 # Update all the information about the hand from the last time to this frame
#                 rem = temp_hand.update_everything(cmh, cmhl)
#
#                 # If hand is successfully tracked and updated, remove from hand results for the frame
#                 # Otherwise remove from list of hands we are tracking
#                 if rem is not None:
#                     cmh.pop(rem)
#                     cmhl.pop(rem)
#                     ind += 1
#                 else:
#                     loh.pop(ind)
#
#             # Add whatever remaining hands that didn't match any of the tracked hands to the tracked hands list
#             # In other words, we begin tracking the "new" hands (might be mistakenly considered new)
#             for index in range(0, len(cmh)):
#                 temp_hand = hand(cmhl[index], cmh[index].classification._values[0].label,
#                                  cmh[index].classification._values[0].score, board)
#                 loh.append(temp_hand)
#             no_hands = 0
#         else:
#             # No hands are detected so remove all the currently tracked hands and update the no_hands state
#             # No hands state only begins keeping track after hands initially appear
#             loh = []
#             if no_hands is not None:
#                 no_hands += 1
#
#         # Draw the points on the hand
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         # For debugging purposes
#         # for h in range(0, len(loh)):
#         #     print(str(h) + ": " + str(loh[h]))
#         #     continue
#
#         # if len(loh) > 0:
#         #     # Test if it's moving
#         #     for val in loh:
#         #         print(val)
#         #     print("     ")
#         #     # if val.moving:
#         #     #     print("moving")
#
#         # Check if the no_hands state has started keeping track and is triggered
#         if no_hands is not None and no_hands > trigger:
#             # If triggered, reset the no hands state and clean up the surface
#
#             # TODO: Might need to remove if lighting is rough
#             # board.surface_level(image)
#             no_hands = 0
#
#         # TODO: Uncomment for testing
#         # image = temp_hand.update_interface(cmh, cmhl, image, cap)
#
#         # Draw circles to mark where blocks were placed in this frame
#         for d in lod:
#             x = int(d[0])
#             y = int(d[1])
#             image = cv2.circle(image, (x, y), 50, (255, 0, 0), 10)
#
#         # Draw grid lines to the image
#         image = drawlines(dsize, image, board)
#
#         # Draw marks on where blocks are located at all times
#         for i in range(0, len(board.top)):
#             for j in range(0, len(board.top[0])):
#                 if board.top[i][j] != 0:
#                     cx, cy = board.centers[i][j]
#                     image = cv2.circle(image, (cx, cy), 30, (0, 255, 0), 10)
#
#         result.write(image)
#
#         # Display the new augmented frame.
#         cv2.imshow('MediaPipe Hands', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
#         frame += 1
#
#     # Close the hand and video code after video or stream is over
#     hands.close()
#     cap.release()
#     result.release()
#
#
# if __name__ == "__main__":
#     main()
