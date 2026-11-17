export type UploadOutcome = {
  kind: 'indexed' | 'already-indexed' | 'unsupported' | 'unreadable' | 'error'
  message: string
}

/** POST /documents and turn the response into user-facing feedback (contract: task 5.4). */
export async function uploadDocument(file: File): Promise<UploadOutcome> {
  const form = new FormData()
  form.append('file', file)
  let response: Response
  try {
    response = await fetch('/documents', { method: 'POST', body: form })
  } catch {
    return { kind: 'error', message: 'Upload failed. Try again.' }
  }
  const body = await response.json().catch(() => ({}))
  switch (response.status) {
    case 201:
      return { kind: 'indexed', message: `Indexed "${body.title}" (${body.chunks} sections).` }
    case 200:
      return { kind: 'already-indexed', message: `"${body.title}" is already indexed.` }
    case 415:
      return { kind: 'unsupported', message: body.detail }
    case 422:
      return { kind: 'unreadable', message: `Couldn't read this file: ${body.detail}` }
    default:
      return { kind: 'error', message: 'Upload failed. Try again.' }
  }
}
