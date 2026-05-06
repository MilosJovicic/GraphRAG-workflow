UNWIND $rows AS row
CALL (row) {
  WITH row WHERE row.label = 'Section'
  MATCH (n:Section)
  WHERE n.id = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'CodeBlock'
  MATCH (n:CodeBlock)
  WHERE n.id = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'TableRow'
  MATCH (n:TableRow)
  WHERE n.id = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'Callout'
  MATCH (n:Callout)
  WHERE n.id = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'Tool'
  MATCH (n:Tool)
  WHERE n.name = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'Hook'
  MATCH (n:Hook)
  WHERE n.name = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'PermissionMode'
  MATCH (n:PermissionMode)
  WHERE n.name = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'MessageType'
  MATCH (n:MessageType)
  WHERE n.name = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'Provider'
  MATCH (n:Provider)
  WHERE n.name = row.node_id
  RETURN n

  UNION
  WITH row WHERE row.label = 'SettingKey'
  MATCH (n:SettingKey)
  WHERE n.key = row.node_id
  RETURN n
}
SET
  n.context         = row.context,
  n.indexed_text    = row.indexed_text,
  n.embedding       = row.embedding,
  n.context_source  = row.context_source,
  n.context_version = row.context_version
RETURN count(n) AS updated
