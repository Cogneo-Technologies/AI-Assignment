search_string = "a crime thriller movie"
minimum_rating = 8.5
# find the movie in the movie.json
# Find if any related general knowledge exists related to the movie.
# If yes, return the movie_name and filename for the gk


def solution(search_string, minimum_rating):
    # Your code goes here

    # Return [("Movie Name", "filename.md"), ...]

    return [  # Sample output
        ("Goodfellas", "tx.md"),
        ("Pulp Fiction", "tx.md"),
        ("The Godfather", "tn.md"),
    ]


def test_solution():
    list_of_movies = solution("a crime thriller movie", 8.7)  # sample input
    # Expected output: [ ("Pulp Fiction", "tx.md") , ("The Godfather", "tn.md") ]]
    # Note the filenames mentioned doesn't exist for the sample input.
    assert any([i[0] == "Goodfellas" for i in list_of_movies])

    list_of_movies = solution("a fantasy movie", 8.5)  # sample input
    # Expected output: [ ("Pulp Fiction", "tx.md") , ("The Godfather", "tn.md") ]]
    # Note the filenames mentioned doesn't exist for the sample input.
    assert any([i[0] == "Sprited Away" for i in list_of_movies])  # This input fails


if __name__ == "__main__":
    test_solution()
    print("All tests passed!")
    # You can add more test cases to validate the solution.
    # For example, you can test with different search strings and check the output.
