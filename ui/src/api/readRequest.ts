const DEFAULT_READ_REQUEST_TIMEOUT_MS = 10_000

const inFlightJsonGets = new Map<string, Promise<unknown>>()

class HttpResponseError extends Error {}

class ReadRequestTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`)
    this.name = 'TimeoutError'
  }
}

/**
 * Read JSON with an in-flight-only GET cache. Concurrent callers for the same
 * URL share one fetch and parsed response; every settled request is removed so
 * later reads always see current server state.
 */
export async function getJson<T>(
  url: string,
  failureMessage: string,
  timeoutMs = DEFAULT_READ_REQUEST_TIMEOUT_MS,
): Promise<T> {
  // Timeout is part of request identity: callers with different bounds should
  // not inherit whichever timeout happened to start first.
  const key = JSON.stringify([url, timeoutMs])
  let request = inFlightJsonGets.get(key)

  if (!request) {
    const controller = new AbortController()
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    const fetchRequest = Promise.resolve()
      .then(() => fetch(url, { signal: controller.signal }))
      .then(response => {
        if (!response.ok) throw new HttpResponseError()
        return response.json()
      })
    const timeoutRequest = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => {
        const error = new ReadRequestTimeoutError(timeoutMs)
        controller.abort(error)
        reject(error)
      }, timeoutMs)
    })

    const pendingRequest: Promise<unknown> = Promise.race([fetchRequest, timeoutRequest]).finally(() => {
      if (timeoutId !== undefined) clearTimeout(timeoutId)
      if (inFlightJsonGets.get(key) === pendingRequest) inFlightJsonGets.delete(key)
    })
    request = pendingRequest
    inFlightJsonGets.set(key, request)
  }

  try {
    return await request as T
  } catch (error) {
    if (error instanceof HttpResponseError || error instanceof ReadRequestTimeoutError) {
      const endpointError = new Error(failureMessage)
      if (error instanceof ReadRequestTimeoutError) endpointError.name = 'TimeoutError'
      throw endpointError
    }
    throw error
  }
}
