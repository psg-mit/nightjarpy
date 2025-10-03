from nightjarpy import LLMConfig, nj_llm_factory


def main(problem, nj_llm) -> float:

    res = nj_llm(
        f"Write the expression to evaluate with `eval` to answer to the math problem:{problem}. Give just the expression."
    )
    ans = eval(res)

    return ans


#### Tests ####

if __name__ == "__main__":
    nj_llm = nj_llm_factory(
        config=LLMConfig(model="openai/gpt-4.1"),
        filename=__name__,
        funcname="main",
    )
    print(
        main(
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            nj_llm,
        )
    )
