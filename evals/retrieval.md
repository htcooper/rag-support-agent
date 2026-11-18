# Retrieval with and without query rewriting

Before: the question is embedded as asked. After: Claude Haiku 5.5 first restates it in FastAPI's vocabulary, and the search uses the question plus the rewrite. Hit@5 counts queries with an expected page in the top 5; MRR (0-1) rewards ranking it higher.

| Queries | n | Hit@5 before | Hit@5 after | MRR before | MRR after |
|---|---|---|---|---|---|
| Should-answer, dev | 22 | 17/22 | 22/22 | 0.62 | 0.95 |
| Should-answer, held-out | 10 | 7/10 | 9/10 | 0.43 | 0.80 |
| Docs' own words | 5 | 5/5 | 5/5 | 0.90 | 0.90 |
| User's words | 5 | 2/5 | 5/5 | 0.25 | 0.87 |
| Vague | 5 | 2/5 | 4/5 | 0.20 | 0.80 |

## Fixed by rewriting (12)

- `cors`: My React app on localhost:3000 gets blocked when it calls my FastAPI backend. How do I fix it? -> *CORS middleware CORSMiddleware allow_origins allow_methods allow_headers cross-origin resource sharing*
- `hide-password`: My API is returning the user's hashed password in the response. How do I stop that? -> *response_model response model exclude hashed password output schema Pydantic model field filtering*
- `timing-header`: How do I add a header to every response showing how long the request took? -> *middleware custom response header timing request process time HTTP middleware @app.middleware("http") call_next Response headers*
- `proxy-prefix`: My app sits behind Nginx under the /api prefix and the interactive docs page is broken. What do I need to configure? -> *FastAPI behind a reverse proxy with root_path, prefix, and proxy headers: docs URLs, openapi_url, root_path, FastAPI root_path parameter, --root-path, --proxy-headers, forwarded-allow-ips, Swagger UI, OpenAPI servers, Behind a Proxy, Nginx path prefix, custom docs_url*
- `big-file`: How do I send a large file to the client without loading the whole thing into memory? -> *StreamingResponse streaming file response generator chunked iter_file async file download*
- `nested-list`: How do I accept an order in the request body that contains a list of items? -> *Request body with nested list of items: Pydantic models, List field type, Body parameter, request body declaration, nested models*
- `admin-only`: How do I make some endpoints readable by admins only? -> *FastAPI security dependencies authorization role-based access control OAuth2 scopes Security HTTPBearer Depends permissions*
- `bg-user`: How do I send an email after returning the response? -> *background tasks BackgroundTasks add_task run after response returned send email*
- `bg-vague`: How do I do something later without making the user wait? -> *background tasks BackgroundTasks run task after response returned*
- `hide-user`: How do I keep the password field out of my API responses? -> *response_model response model exclude password field output data filtering Pydantic*
- `db-user`: How do I make sure my database connection closes after each request? -> *FastAPI dependencies with yield, database session cleanup, dependency teardown after response, lifespan events, Depends*
- `db-vague`: How do I clean up after a request? -> *dependencies with yield, exit code, cleanup after response, lifespan events, background tasks, middleware*

## Broken by rewriting (0)

None.

Cost of this comparison: $0.0015.
