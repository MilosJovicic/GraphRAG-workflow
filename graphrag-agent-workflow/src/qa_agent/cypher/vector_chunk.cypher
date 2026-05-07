CALL db.index.vector.queryNodes('chunk_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section {id: c.section_id})
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  c.id            AS node_id,
  'Chunk'         AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
