/** What Save does with the entered title: an empty or unchanged one just closes the form. */
export function saveAction(current: string, entered: string): { kind: 'cancel' } | { kind: 'save'; title: string } {
  const title = entered.trim()
  return title === '' || title === current ? { kind: 'cancel' } : { kind: 'save', title }
}
