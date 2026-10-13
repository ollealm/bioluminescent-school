/**
 * The "document": a small paper icon top-right opens a floating, transparent
 * note with thin borders that holds the portfolio links. It can be dragged
 * anywhere (pointer events, so touch works too) and closed. Markup and
 * styles live in index.html; this file only wires behavior.
 *
 * The link list is a placeholder set: Olle fills in the five pages.
 */
export function installDoc(): void {
  const toggle = document.getElementById("doc-toggle");
  const doc = document.getElementById("doc");
  const close = document.getElementById("doc-close");
  if (!toggle || !doc || !close) return;

  const show = (on: boolean): void => {
    doc.hidden = !on;
    toggle.setAttribute("aria-expanded", String(on));
  };
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    show(doc.hidden);
  });
  close.addEventListener("click", (e) => {
    e.stopPropagation();
    show(false);
  });

  // Drag by any part of the document except its links and the close
  // button. Position is kept as a translate so the CSS anchor stays.
  let dx = 0;
  let dy = 0;
  let drag: { id: number; x: number; y: number; ox: number; oy: number } | null = null;
  doc.addEventListener("pointerdown", (e) => {
    const t = e.target as HTMLElement;
    if (t.closest("a, button")) return;
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, ox: dx, oy: dy };
    doc.setPointerCapture(e.pointerId);
    doc.classList.add("dragging");
    e.preventDefault();
  });
  doc.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    dx = drag.ox + (e.clientX - drag.x);
    dy = drag.oy + (e.clientY - drag.y);
    doc.style.transform = `translate(${dx}px, ${dy}px)`;
    // While dragging the note, the pointer is not a predator.
    e.stopPropagation();
  });
  const end = (e: PointerEvent): void => {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null;
    doc.classList.remove("dragging");
  };
  doc.addEventListener("pointerup", end);
  doc.addEventListener("pointercancel", end);
}
