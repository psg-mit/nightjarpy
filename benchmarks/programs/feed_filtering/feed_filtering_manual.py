from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from nightjarpy import nj_llm_factory


class FeedFilterResult(BaseModel):
    """Structured output for filtered posts."""

    posts: list[str] = Field(description="Filtered list of posts that adhere to the user's feed preference.")


def main(posts: list[str], feed_preference: str, nj_llm):
    result: FeedFilterResult = nj_llm(
        "Remove posts from <posts> that do not adhere to the user's feed preference "
        "<feed_preference> and store the result in `posts`.\n"
        f"<posts>{posts}</posts><feed_preference>{feed_preference}</feed_preference>",
        output_format=FeedFilterResult,
    )

    return result.posts


#### Tests ####


def run(
    nj_llm,
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:
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

    outputs = {}
    errors = {}
    hard_results = {
        "test_technical_output_type": False,
        "test_technical_output_content_1": False,
        "test_technical_output_content_2": False,
        "test_technical_output_content_3": False,
        "test_technical_output_content_4": False,
        "test_technical_output_content_5": False,
        "test_spicy_output_type": False,
        "test_spicy_output_content_1": False,
        "test_spicy_output_content_2": False,
        "test_spicy_output_content_3": False,
        "test_spicy_output_content_4": False,
        "test_spicy_output_content_5": False,
    }

    try:
        technical_new_posts = main(POSTS.copy(), TECHNICAL_FEED_PREFERENCE, nj_llm)
        outputs["test_technical"] = (
            POSTS,
            TECHNICAL_FEED_PREFERENCE,
            technical_new_posts,
        )
    except Exception as e:
        errors["test_technical"] = e
    else:
        hard_results["test_technical_output_type"] = isinstance(technical_new_posts, list)
        if hard_results["test_technical_output_type"]:
            hard_results["test_technical_output_content_1"] = POSTS[0] in technical_new_posts
            hard_results["test_technical_output_content_2"] = POSTS[1] not in technical_new_posts
            hard_results["test_technical_output_content_3"] = POSTS[2] in technical_new_posts
            hard_results["test_technical_output_content_4"] = POSTS[3] not in technical_new_posts
            hard_results["test_technical_output_content_5"] = POSTS[4] in technical_new_posts

    try:
        spicy_new_posts = main(POSTS.copy(), SPICY_FEED_PREFERENCE, nj_llm)
        outputs["test_spicy"] = (POSTS, SPICY_FEED_PREFERENCE, spicy_new_posts)
    except Exception as e:
        errors["test_spicy"] = e
    else:
        hard_results["test_spicy_output_type"] = isinstance(spicy_new_posts, list)
        if hard_results["test_spicy_output_type"]:
            hard_results["test_spicy_output_content_1"] = POSTS[0] not in spicy_new_posts
            hard_results["test_spicy_output_content_2"] = POSTS[1] in spicy_new_posts
            hard_results["test_spicy_output_content_3"] = POSTS[2] not in spicy_new_posts
            hard_results["test_spicy_output_content_4"] = POSTS[3] in spicy_new_posts
            hard_results["test_spicy_output_content_5"] = POSTS[4] not in spicy_new_posts

    return outputs, errors, hard_results


if __name__ == "__main__":
    from nightjarpy.configs import LLMConfig

    config = LLMConfig()
    nj_llm = nj_llm_factory(config=config, filename=__file__, funcname="main", max_calls=100)
    results, errors, hard_results = run(nj_llm)
    print(results)
    print(hard_results)
    print(errors)
