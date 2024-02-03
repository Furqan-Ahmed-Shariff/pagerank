import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    df_complement = 1 - damping_factor
    p_rand_pagesel = df_complement / len(corpus)
    t_model = dict()
    n_links = len(corpus[page])

    for page_link in corpus[page]:
        t_model[page_link] = damping_factor / n_links

    for page in corpus:
        t_model[page] = t_model.get(page, 0) + p_rand_pagesel

    return t_model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = dict()
    page = random.choice(list(corpus.keys()))
    # t_model = transition_model(corpus, page, damping_factor)
    # samples = random.choices(list(t_model.keys()), weights=list(t_model.values()), k=1)

    for i in range(n):
        pageRank[page] = pageRank.get(page, 0) + 1
        # page = random.choice(samples)
        t_model = transition_model(corpus, page, damping_factor)
        page = random.choices(
            list(t_model.keys()), weights=list(t_model.values()), k=1
        )
        page = page[0]

    for key in pageRank:
        pageRank[key] /= n
    return pageRank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = dict()
    pageRankNew = dict()
    nlinks = dict()
    corpusLen = len(corpus)
    for page in corpus:
        pageRankNew[page] = 1 / corpusLen
        nlinks[page] = len(corpus[page])
        
    flag = True
    while(flag):
        flag = False
        pageRank = pageRankNew.copy()
        for page in corpus:
            pr_page = (1 - damping_factor) / corpusLen
            sum = 0
            for other_page in corpus:
                if page in corpus[other_page] and nlinks[other_page] != 0:
                    sum += pageRank[other_page] / nlinks[other_page]
                elif nlinks[other_page] == 0:
                    sum += pageRank[other_page] / corpusLen

            sum *= damping_factor
            pr_page += sum
            pageRankNew[page] = pr_page
            if abs(pageRank[page] - pageRankNew[page]) > 0.001:
                flag = True
    return pageRankNew


if __name__ == "__main__":
    main()
