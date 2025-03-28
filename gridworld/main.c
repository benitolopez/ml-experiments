/*
 * Q-Learning implementation for a simple GridWorld environment in C.
 *
 * The agent learns the optimal path from a starting point to a goal,
 * avoiding obstacles by updating Q-values based on exploration and
 * exploitation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Grid dimensions
#define GRID_SIZE 5

// Q-Learning parameters
#define LEARNING_RATE 0.1
#define DISCOUNT_FACTOR 0.9

// Representation of the grid environment
// -10: obstacle, 10: goal, 0: empty cell
int grid[GRID_SIZE][GRID_SIZE] = {{0, 0, 0, -10, 10},
                                  {0, -10, -10, 0, 0},
                                  {0, 0, 0, 0, 0},
                                  {0, 0, -10, 0, 0},
                                  {0, 0, 0, 0, 0}};

// Rewards associated with each cell
// Goal provides high positive reward, obstacles large negative, others small
// negative to encourage shortest path learning
double rewards[GRID_SIZE][GRID_SIZE] = {{-1, -1, -1, -10, 10},
                                        {-1, -10, -10, -1, -1},
                                        {-1, -1, -1, -1, -1},
                                        {-1, -1, -10, -1, -1},
                                        {-1, -1, -1, -1, -1}};

// Represents the agent's position on the grid
typedef struct {
  int x, y;
} Agent;

// Movement directions (UP, DOWN, LEFT, RIGHT)
int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

// Check if a given position is valid (within bounds and not an obstacle)
int is_valid_move(int x, int y, int grid[GRID_SIZE][GRID_SIZE]) {
  if (x < 0 || y < 0 || x >= GRID_SIZE || y >= GRID_SIZE)
    return 0;

  if (grid[x][y] == -10)
    return 0;

  return 1;
}

// Q-table storing Q-values for each state-action pair
double Q[GRID_SIZE][GRID_SIZE][4] = {0.0};

// Update the Q-value based on the Q-learning algorithm
void update_q_value(int x, int y, int action, int new_x, int new_y) {
  double reward = rewards[new_x][new_y];

  // Identify the maximum future Q-value for the new state
  double max_next_q = Q[new_x][new_y][0];
  for (int a = 1; a < 4; a++) {
    if (Q[new_x][new_y][a] > max_next_q) {
      max_next_q = Q[new_x][new_y][a];
    }
  }

  // Update Q-value using the Q-learning update rule
  Q[x][y][action] +=
      LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - Q[x][y][action]);
}

// Check if a specific action from position (x,y) is valid
int is_valid_action(int x, int y, int action, int grid[GRID_SIZE][GRID_SIZE]) {
  int new_x = x + dx[action];
  int new_y = y + dy[action];

  return is_valid_move(new_x, new_y, grid);
}

// Select an action using epsilon-greedy policy
int select_action(int x, int y, double epsilon,
                  int grid[GRID_SIZE][GRID_SIZE]) {
  double r = (double)rand() / RAND_MAX;

  int valid_actions[4];
  int valid_count = 0;

  // Collect all valid actions from current state
  for (int a = 0; a < 4; a++) {
    if (is_valid_action(x, y, a, grid)) {
      valid_actions[valid_count++] = a;
    }
  }

  // If no valid actions available, return -1 (agent is stuck)
  if (valid_count == 0)
    return -1;

  if (r < epsilon) {
    // Exploration: choose a random valid action
    return valid_actions[rand() % valid_count];
  } else {
    // Exploitation: choose the best known valid action
    int best_action = valid_actions[0];
    double max_q = Q[x][y][best_action];

    for (int i = 1; i < valid_count; i++) {
      int a = valid_actions[i];
      if (Q[x][y][a] > max_q) {
        max_q = Q[x][y][a];
        best_action = a;
      }
    }

    return best_action;
  }
}

int main() {
  srand(time(NULL));

  int NUM_EPISODES = 10000;
  int MAX_STEPS = 100;

  double epsilon = 1.0;
  double epsilon_decay = 0.999;

  // Training phase: Run episodes to learn Q-values
  for (int episode = 0; episode < NUM_EPISODES; episode++) {
    Agent agent = {0, 0};

    for (int step = 0; step < MAX_STEPS; step++) {
      epsilon *= epsilon_decay;

      int action = select_action(agent.x, agent.y, epsilon, grid);
      if (action == -1)
        break; // Agent stuck, end episode

      int new_x = agent.x + dx[action];
      int new_y = agent.y + dy[action];

      update_q_value(agent.x, agent.y, action, new_x, new_y);
      agent.x = new_x;
      agent.y = new_y;

      if (grid[new_x][new_y] == 10)
        break; // Goal reached
    }

    if ((episode + 1) % 1000 == 0)
      printf("Episode %d completed.\n", episode + 1);
  }

  // Test the learned policy by visualizing the path taken
  char path_grid[GRID_SIZE][GRID_SIZE];
  for (int i = 0; i < GRID_SIZE; i++)
    for (int j = 0; j < GRID_SIZE; j++)
      path_grid[i][j] = (grid[i][j] == -10)  ? 'X'
                        : (grid[i][j] == 10) ? 'G'
                                             : '.';

  path_grid[0][0] = 'S';

  Agent agent = {0, 0};
  for (int step = 0; step < MAX_STEPS; step++) {
    int action = select_action(agent.x, agent.y, 0.0, grid);
    if (action == -1 || !is_valid_action(agent.x, agent.y, action, grid)) {
      printf("Invalid move encountered.\n");
      break;
    }

    agent.x += dx[action];
    agent.y += dy[action];

    if (path_grid[agent.x][agent.y] == '.')
      path_grid[agent.x][agent.y] = '*';

    if (grid[agent.x][agent.y] == 10) {
      printf("Goal reached!\n");
      break;
    }
  }

  printf("\nAgent's path:\n");
  for (int i = 0; i < GRID_SIZE; i++) {
    for (int j = 0; j < GRID_SIZE; j++)
      printf("%c ", path_grid[i][j]);
    printf("\n");
  }

  return 0;
}
