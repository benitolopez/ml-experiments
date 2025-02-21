#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 150
#define MAX_NAMES 32033
#define MAX_NAME_LEN 35
#define ALPHABET_SIZE 28
#define MIN_NAME_LENGTH 3
#define MAX_NAME_LENGTH 12
#define PROBABILITY_THRESHOLD 0.01

int char_to_index(char c) {
  if (c == '^') {
    return 26; // Start marker
  }
  if (c == '$') {
    return 27; // End marker
  }
  if (c >= 'a' && c <= 'z') {
    return c - 'a'; // Letters a-z
  }
  return -1; // Invalid character
}

char index_to_char(int index) {
  if (index >= 0 && index < 26) {
    return 'a' + index; // Convert 0-25 back to 'a'-'z'
  } else if (index == 26) {
    return '^'; // Start token
  } else if (index == 27) {
    return '$'; // End token
  } else {
    return '?'; // Invalid index
  }
}

int validate_index(int key, char c) {
  if (key < 0 || key >= ALPHABET_SIZE) {
    printf("Invalid index %d for char '%c'\n", key, c);
    return 0;
  }
  return 1;
}

// Helper function to check if character is a vowel
int is_vowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
}

// Check if a letter combination is valid
int is_valid_combination(char prev, char curr) {
  // Prevent double consonants at start
  if (prev == '^' && !is_vowel(curr)) {
    const char *valid_starts = "bcdfghjklmnprstvw"; // Common name starts
    return strchr(valid_starts, curr) != NULL;
  }

  // Prevent triple consonants
  if (!is_vowel(prev) && !is_vowel(curr)) {
    return strchr("lr", curr) != NULL; // Allow only 'l' or 'r' after consonant
  }

  return 1;
}

double select_next_char(double probabilities[], char current_char, char *name,
                        int name_len, int *selected_index) {
  double cumulative_probs[ALPHABET_SIZE] = {0};
  double probs_sum = 0.0;

  // Calculate cumulative probabilities for valid transitions only
  for (int i = 0; i < ALPHABET_SIZE; i++) {
    char next_char = index_to_char(i);

    // Skip if probability is too low
    if (probabilities[i] < PROBABILITY_THRESHOLD) {
      continue;
    }

    // Skip invalid combinations
    if (!is_valid_combination(current_char, next_char)) {
      continue;
    }

    // Count consecutive vowels/consonants leading up to current position
    int consecutive_vowels = 0;
    int consecutive_consonants = 0;
    for (int j = name_len - 1; j >= 0 && j >= name_len - 2; j--) {
      if (is_vowel(name[j])) {
        consecutive_vowels++;
        consecutive_consonants = 0;
      } else {
        consecutive_consonants++;
        consecutive_vowels = 0;
      }
    }

    // Prevent triple vowels
    if (consecutive_vowels >= 2 && is_vowel(next_char)) {
      continue;
    }

    // Prevent triple consonants
    if (consecutive_consonants >= 2 && !is_vowel(next_char)) {
      continue;
    }

    probs_sum += probabilities[i];
    cumulative_probs[i] = probs_sum;
  }

  // If no valid transitions, force end of name
  if (probs_sum < PROBABILITY_THRESHOLD) {
    *selected_index = char_to_index('$');
    return 0.0;
  }

  double r = ((double)rand() / RAND_MAX) * probs_sum;

  for (int i = 0; i < ALPHABET_SIZE; i++) {
    if (cumulative_probs[i] > 0 && r <= cumulative_probs[i]) {
      *selected_index = i;
      return probabilities[i];
    }
  }

  *selected_index = char_to_index('$');
  return 0.0;
}

int main() {
  // Use a better seed
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  srand(ts.tv_nsec);

  FILE *file = fopen("data/names.txt", "r");
  if (file == NULL) {
    printf("Error: Cannot read the dataset file.\n");
    return 1;
  }

  char line[MAX_NAME_LEN];
  char dataset[MAX_NAMES][MAX_NAME_LEN];

  int current_row = 0;

  // Read the sample file and store the data
  while (fgets(line, sizeof(line), file) != NULL && current_row < MAX_NAMES) {
    // Convert name to lowercase
    for (int i = 0; line[i]; i++) {
      line[i] = tolower(line[i]);
    }

    // Remove the newline character if present
    line[strcspn(line, "\n")] = '\0';

    // Wrap names with ^^ at the start and $ at the end
    snprintf(dataset[current_row], MAX_NAME_LEN, "^^%s$", line);

    current_row++;
  }

  fclose(file);

  int trigram_counts[ALPHABET_SIZE][ALPHABET_SIZE][ALPHABET_SIZE] = {0};

  for (int i = 0; i < current_row; i++) {
    char *name = dataset[i];
    int name_length = strlen(name);

    for (int j = 0; j < name_length - 2; j++) {
      int key_first = char_to_index(name[j]);
      int key_second = char_to_index(name[j + 1]);
      int key_third = char_to_index(name[j + 2]);

      if (validate_index(key_first, name[j]) &&
          validate_index(key_second, name[j + 1]) &&
          validate_index(key_third, name[j + 2])) {
        trigram_counts[key_first][key_second][key_third]++;
      }
    }
  }

  double trigram_probs[ALPHABET_SIZE][ALPHABET_SIZE][ALPHABET_SIZE] = {0};

  for (int i = 0; i < ALPHABET_SIZE; i++) {
    for (int j = 0; j < ALPHABET_SIZE; j++) {
      int total_count = 0;

      for (int n = 0; n < ALPHABET_SIZE; n++) {
        total_count += trigram_counts[i][j][n];
      }

      // Avoid division by zero
      if (total_count == 0) {
        continue;
      }

      for (int k = 0; k < ALPHABET_SIZE; k++) {
        trigram_probs[i][j][k] = (double)trigram_counts[i][j][k] / total_count;
      }
    }
  }

  char generated_name[MAX_NAME_LEN];

  int index = 0;

  for (int name_count = 0; name_count < 10; name_count++) {
    char generated_name[MAX_NAME_LEN];
    int index = 0;
    int first_char_index = char_to_index('^');
    int second_char_index = char_to_index('^');

    while (index < MAX_NAME_LENGTH) {
      int next_char_index;
      double prob =
          select_next_char(trigram_probs[first_char_index][second_char_index],
                           index_to_char(second_char_index), generated_name,
                           index, &next_char_index);

      // End generation if we hit the end token
      if (next_char_index == char_to_index('$')) {
        if (index >= MIN_NAME_LENGTH) {
          break;
        } else {
          // Reset and try again if name is too short
          index = 0;
          first_char_index = char_to_index('^');
          second_char_index = char_to_index('^');
          continue;
        }
      }

      generated_name[index] = index_to_char(next_char_index);

      // Update indices for next iteration - slide the window
      first_char_index = second_char_index;
      second_char_index = next_char_index;

      index++;

      // Force end if we're getting too long
      if (index >= MAX_NAME_LENGTH - 1) {
        break;
      }
    }

    generated_name[index] = '\0';

    // Additional validation of final name
    int valid = 1;
    if (strlen(generated_name) < MIN_NAME_LENGTH)
      valid = 0;
    if (!is_vowel(generated_name[0]) && !is_vowel(generated_name[1]))
      valid = 0;

    // Retry if invalid
    if (!valid) {
      name_count--;
      continue;
    }

    printf("Generated Name %d: %s\n", name_count + 1, generated_name);
  }

  return 0;
}
