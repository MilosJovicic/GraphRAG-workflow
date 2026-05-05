CALL db.index.vector.queryNodes('section_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS s, score
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  s.id          AS node_id,
  'Section'     AS node_label,
  s.indexed_text AS indexed_text,
  s.text        AS raw_text,
  p.url         AS url,
  s.anchor      AS anchor,
  s.breadcrumb  AS breadcrumb,
  s.breadcrumb  AS title,
  score         AS vector_score
ORDER BY score DESC
LIMIT $limit
