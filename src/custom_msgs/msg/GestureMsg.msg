uint8 ROCK=0
uint8 PAPER=1
uint8 SCISSORS=2
uint8 NONE=255

bool detected

uint8 player_gesture      # enum-like
float32 confidence # from CNN
float32 x_min
float32 y_min
float32 x_max
float32 y_max

float32[] landmarks_x
float32[] landmarks_y

int32 global_idx