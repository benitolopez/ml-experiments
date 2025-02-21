# Name Generator using Trigrams

This folder contains a character-level trigram model implemented from scratch in C for generating names. The model learns transition probabilities between sequences of three characters to generate new names that follow similar patterns to the training data.

## The Trigram Model

A trigram model is a simple statistical language model that predicts the next character based on the previous two characters. For example, for the name "MARIA":

| Context | Next Char | Example |
| ------- | --------- | ------- |
| ^^      | M         | Start of name |
| ^M      | A         | First real transition |
| MA      | R         | Second transition |
| AR      | I         | Third transition |
| RI      | A         | Fourth transition |
| IA      | $         | End of name |

## Model Architecture
- Input: Two previous characters (using ^ as start token, $ as end token)
- Transition Matrix: 28x28x28 (26 letters + 2 special tokens)
- Probability Calculation: P(next_char|prev_two_chars)
- Generation Method: Probabilistic sampling with linguistic constraints

## Constraints
- Minimum name length: 3 characters
- Maximum name length: 12 characters
- Prevents triple consonants (except with 'l' or 'r')
- Prevents triple vowels
- Controls valid starting consonants
- Filters low-probability transitions

## Results

The model generates names that follow basic linguistic patterns but shows clear limitations:

Good examples:
- savya
- royce
- noriana
- alana
- myke

Issues:
- Sometimes produces overly long names (catalayleen, samerigarie)
- Can create unnatural letter combinations (eigoreemerr)
- Occasionally generates implausible vowel sequences

The trigram model shows better results than simpler bigram approaches, but still has clear limitations. While it can generate some natural-sounding names like "savya" and "royce", it also produces unrealistic ones like "catalayleen" and "eigoreemerr". This suggests that name generation might benefit from more sophisticated approaches like neural networks.
