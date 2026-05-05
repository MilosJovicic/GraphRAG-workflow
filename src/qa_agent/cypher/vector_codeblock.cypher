CALL db.index.vector.queryNodes('codeblock_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS b, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(b)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($language IS NULL OR b.language = $language)
RETURN
  b.id            AS node_id,
  'CodeBlock'     AS node_label,
  b.indexed_text  AS indexed_text,
  b.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  COALESCE('[' + b.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
