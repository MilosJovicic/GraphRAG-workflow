UNWIND $seeds AS seed
MATCH (parent:Section)-[r]->(seed_node)
WHERE seed_node.id = seed.node_id
  AND any(lbl IN labels(seed_node) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow'])
WITH parent, seed.node_id AS seed_id
MATCH (parent)-[]->(sibling)
WHERE sibling.id IS NOT NULL
  AND sibling.id <> seed_id
  AND any(lbl IN labels(sibling) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow'])
WITH seed_id, sibling, head([lbl IN labels(sibling) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow']]) AS sibling_label
RETURN
  sibling.id           AS node_id,
  sibling_label        AS node_label,
  sibling.indexed_text AS indexed_text,
  sibling.text         AS raw_text,
  null                 AS url,
  null                 AS anchor,
  null                 AS breadcrumb,
  null                 AS title,
  seed_id              AS seed_id
LIMIT $cap
