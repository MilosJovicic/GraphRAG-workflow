UNWIND $ids AS id
OPTIONAL MATCH (n) WHERE n.id = id OR n.url = id OR n.name = id OR n.key = id
WITH id, n
WHERE n IS NOT NULL
RETURN
  id                   AS node_id,
  head(labels(n))      AS node_label,
  n.indexed_text       AS indexed_text,
  COALESCE(n.text, n.description, n.title, n.name) AS raw_text
