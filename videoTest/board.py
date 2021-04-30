from videoTest import util

###############
# DEFINITIONS #
###############

# drop point: preset IRL coordinates that determine where a block is placed. block's location is based on the closest
# drop point the block is located at

# index set: the indices that the drop point has in the 2D array. Maps to point that tells Minecraft where to
# build (pretend the building space in Minecraft is a grid)

#########################
# ADJUSTABLE PARAMETERS #
#########################

# self.side_length
# self.side_deviation_threshold
# self.border_ratio
# function get_contours – every param used in the functions


class Board:

    def __init__(self, img):
        self.side_length = 0

        # The side length of the contour can be up to (self.side_deviation_threshold * 100)% smaller than the side
        # length
        self.side_deviation_threshold = 0.8

        # Defines the border of the board (self.board / self.border_ratio)
        # Anything found at the border is assumed to be held and ready to be placed rather than actually part of the
        # structure
        self.border_ratio = 6

        self.set_side_length(img)

        if self.side_length is None:
            self.side_length = 30

        # TODO: Remove when standard height is set and workflow for finding side length occurs
        self.side_length = 30

        self.width, self.height = img.shape[0], img.shape[1]

        # Creates an array with all the possible "drop points" on the board
        # Whatever is placed on the board will map to the closest center, to standardize where the blocks will map to in
        # Minecraft, which are the index set for top

        self.centers = [[(self.side_length // 2 + j * self.side_length, # min(self.height, self.side_length // 2 + j * self.side_length),
                          self.side_length // 2 + i * self.side_length) # min(self.width, self.side_length // 2 + i * self.side_length))
                         for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]

        # Creates an array that represents a topological graph.
        # Each point represents the height (in blocks) at that point
        self.top = [[0 for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]

        # Array that checks if a block is there. If it is not, yet the topological array says there is, then the block
        # gets removed

        # Essentially a helpful heuristic in case the grab detector didn't pick up on the removed block
        self.there = [[False for j in range(self.height // self.side_length + 1)] for l in range(self.width // self.side_length + 1)]

        self.send_info = {}

    def set_side_length(self, img):
        self.side_length = util.get_side_length(img)

    # Convert top corner coordinate to center coordinate
    def tc_to_center(self, x, y, w, h):
        return (x + (w // 2)), (y + (h // 2))

    # Get the index set mapped to the closest "drop point" based on (x, y) coordinates
    def get_center(self, x, y):
        # Update the board when new blocks come up
        for p in range(len(self.centers)):
            for q in range(len(self.centers[0])):
                # Determine where the block is based on which center it's closest to
                # Center is like a "drop point" -- see doc
                center = self.centers[p][q]
                x_diff = abs(center[0] - x)
                y_diff = abs(center[1] - y)

                # If within half of the width of the bounding box, then it is the closest point
                if x_diff <= self.side_length // 2 and y_diff <= self.side_length // 2:
                    return p, q

    # Check if a block doesn't exist at a index set
    def is_block_not_at_center(self, x, y):
        # Map coordinates to a index set
        p, q = self.get_center(x, y)

        if (self.top[p][q] == 0) != (p not in self.send_info.keys() or q not in self.send_info[p].keys()):
            print("Send info implementation is not correct")

        # If block doesn't exist at the index set
        # 0 means no blocks at that point b/c height is zero
        return self.top[p][q] == 0

    # Remove single block at given coordinates
    def remove_single(self, x, y):
        # Map coordinates to a index set
        p, q = self.get_center(x, y)

        # Check if there even is a block at that index set
        if self.is_block_not_at_center(x, y):
            print("Error: Removing something that isn't there.")
        else:
            self.top[p][q] -= 1
            z = len(list(self.send_info[p][q].keys())) - 1
            del self.send_info[p][q][z]
            if len(list(self.send_info[p][q].keys())) == 0:
                del self.send_info[p][q]
            if len(list(self.send_info[p].keys())) == 0:
                del self.send_info[p]

    def add_single_json(self, p, q):
        if p not in self.send_info.keys():
            self.send_info[p] = {}
        if q not in self.send_info[p].keys():
            self.send_info[p][q] = {}
        z = len(list(self.send_info[p][q].keys()))
        self.send_info[p][q][z] = "wood"

    def invert(self):
        res = {}
        for i in self.send_info.keys():
            for j in self.send_info[i].keys():
                for k in self.send_info[j].keys():
                    mat = self.send_info[i][j][k]
                    res[i] = res.get(i, {k: {j: mat}})
                    res[i][k] = res[i].get(k, {j: mat})
                    res[i][k][j] = mat
        self.send_info = res

    # Add a single block at given coordinates
    def add_single(self, x, y, low_layer=False):
        # Map coordinates to that index set
        p, q = self.get_center(x, y)

        # low_layer is a flag to see where the first layer of blocks is at (height of at least 1)
        # you don't really want to change anything if the flag is on because it's just trying to read the contours
        # the exception is that it serves as another layer of check in case grab detection doesn't detect something
        # placed at a point
        # "we know at least one thing is there" -- basically a heuristic
        if not low_layer:
            self.top[p][q] += 1
            self.add_single_json(p, q)
        elif self.top[p][q] == 0:
            self.top[p][q] += 1
            self.add_single_json(p, q)

        # it's trying to read the contours so that we can clear out any blocks that have been mistakenly placed or not
        # detected as removed when it actually was removed
        if low_layer:
            # There was a block there, so don't remove when we check later.
            self.there[p][q] = True

    # Function to use if user decides when to build
    def build_activated(self, log, img):
        # Checks the log file as to where and what operations needs to be done (grab block, release block)
        for x, y, release in log:
            if release:
                self.remove_single(x, y)
            else:
                self.add_single(x, y)
