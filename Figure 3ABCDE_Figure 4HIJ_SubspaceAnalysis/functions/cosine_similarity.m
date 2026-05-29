% Function to compute cosine similarity between two vectors
function cos_sim = cosine_similarity(a, b)
    cos_sim = dot(a, b) / (norm(a) * norm(b));
end
