import string

word = "apple"
guesses = 8
used = []

print("Welcome to the game Hangman!")
print(f"I am thinking of a word that is {len(word)} letters long.")
print("-----------")

while guesses > 0:
    display = ""
    for c in word:
        if c in used:
            display += c
        else:
            display += "_"

    if display == word:
        print("Congratulations, you won!")
        break

    print(f"You have {guesses} guesses left.")
    print("Available Letters:", "".join([c for c in string.ascii_lowercase if c not in used]))

    letter = input("Please guess a letter: ").lower()

    if letter in used:
        print("Oops! You've already guessed that letter:", display)
    else:
        used.append(letter)

        if letter in word:
            print("Good guess:", display)
        else:
            print("Oops! That letter is not in my word:", display)
            guesses -= 1

    print("-----------")

if guesses == 0:
    print("Sorry, you ran out of guesses. The word was:", word)