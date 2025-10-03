import marvin


@marvin.fn
def filter_posts(posts: list[str], feed_preference: str) -> list[str]:
    """
    Remove posts from `posts` that do not adhere to the user's feed preference `feed_preference`
    """


def main(posts: list[str], feed_preference: str) -> list[str]:
    posts = filter_posts(posts, feed_preference)
    return posts


#### Tests ####


def run() -> list[tuple]:
    """
    Runs the main function and returns a list of tuples of input and output.
    """
    POSTS = [
        "How is it that an LLM can know more than any single human but be unable to generate novel scientific discoveries?",
        "AI is not useful for anything. I will die on this hill.",
        "There's a serious question of what happens to LLM scaling when we run out of new high-quality data sources.",
        "Next-token prediction just doesn't make sense as a path to AGI. Why are we even funding this?",
        "Did you know a beaver's teeth never stop growing? They need to gnaw on wood constantly to keep them from growing too long.",
    ]

    TECHNICAL_FEED_PREFERENCE = """
        I like content that discusses AI from a technical or philosophical perspective.
        I am fine with content that is not about AI, so long as it is informative.
        I do not like content that is deliberately inflammatory.
    """
    SPICY_FEED_PREFERENCE = """
        I don't like dry, technical content. I just want some hot takes and ragebait.
        I'm also scared of beavers. Don't mention them.
    """
    technical_new_posts = main(POSTS, TECHNICAL_FEED_PREFERENCE)
    spicy_new_posts = main(POSTS, SPICY_FEED_PREFERENCE)
    return [
        ((POSTS, TECHNICAL_FEED_PREFERENCE), technical_new_posts),
        ((POSTS, SPICY_FEED_PREFERENCE), spicy_new_posts),
    ]


def tests(input_output: list[tuple]) -> list[bool]:
    """
    Verifies that the output of main() is as expected:
      - The output is a list.
      - The output list contains only posts that adhere to the user's feed preference.
    """
    res = []
    res.append(all(isinstance(posts, list) for _, posts in input_output))
    technical_new_posts = input_output[0][1]
    spicy_new_posts = input_output[1][1]
    posts = input_output[0][0][0]
    res.append(posts[1] not in technical_new_posts and posts[3] not in technical_new_posts)
    res.append(posts[0] not in spicy_new_posts and posts[2] not in spicy_new_posts and posts[4] not in spicy_new_posts)
    return res


def output_to_llm_judge(input_output: list[tuple]) -> str:
    posts = input_output[0][0][0]
    TECHNICAL_FEED_PREFERENCE = input_output[0][0][1]
    SPICY_FEED_PREFERENCE = input_output[1][0][1]
    technical_new_posts = input_output[0][1]
    spicy_new_posts = input_output[1][1]

    res = f"=== Start Test ===\n\nOriginal Posts: {posts}\n"

    res += f"Technical Feed Preference: {TECHNICAL_FEED_PREFERENCE}\n"
    res += f"Feed Filtered Based on Technical Preference: {technical_new_posts}\n"
    res += f"=== End Test ===\n\n"

    res += f"=== Start Test ===\n\nOriginal Posts: {posts}\n"
    res += f"Spicy Feed Preference: {SPICY_FEED_PREFERENCE}\n"
    res += f"Feed Filtered Based on Spicy Preference: {spicy_new_posts}\n"

    res += f"=== End Test ===\n\n"
    return res


if __name__ == "__main__":
    input_output = run()
    print(input_output)
    print(tests(input_output))
    print(output_to_llm_judge(input_output))
