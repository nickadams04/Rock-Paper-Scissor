uint8 ROCK=0
uint8 PAPER=1
uint8 SCISSORS=2
uint8 NONE=3

bool detected

uint8 player_gesture      # enum-like
float32 confidence # from CNN
int32 x_min
int32 y_min
int32 x_max
int32 y_max

int32[] landmarks_x
int32[] landmarks_y

int32 global_idx