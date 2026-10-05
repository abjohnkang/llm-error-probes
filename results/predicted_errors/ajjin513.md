# Predicted Error: Hong Ju Jin

Copy this file to `<your-github-username>.md` and fill it in with one question you think Qwen2.5-3B will get wrong.

**Question:** How many times does the letter "r" appear in the word "strawberry"?

- A. 1
- B. 2
- C. 3
- D. 4

**Correct answer:** C

**I predict the model picks:** B

**Why:** The model reads text as tokens (chunks like "straw" + "berry"), not individual letters, so counting letters is hard for it.
