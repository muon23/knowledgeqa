# Knowledge Q&A

Allowing one to ask questions to a set of existing knowledge base.

## Project Structure

- **notebooks**: Contains Jupyter notebooks for experimentation
- **src/main**: Contains the code
  - **utilities**: Useful utilities copied from my other projects
    - **llm**: Wrappers around LLM model accesses
    - **embedding**: Wrappers around text embedding packages
    - **db**: Wrappers for database accesses
  - **knowldgeqa**: Contains code for this assignment
    - **api**: REST API access
    - **bots**: Chatbots that answer questions
    - **indexer**: Indexing the known knowledge for the chatbot to retrieve

- **src/test**: Contains the tests for the code
  - **cjw/knowledgeqa/bots/Evaluate**: To evaluate the effectiveness of the answer.
- **data**: Contains test data
- **deployment**: Contains scripts for deployment
