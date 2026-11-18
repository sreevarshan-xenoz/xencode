/** A widget that renders a label. */
import { helper } from "./helper";
import type { Config } from "config";
import "side-effect-polyfill";

export interface WidgetOptions {
  label: string;
  /** Called when the widget is clicked. */
  onClick(value: string): void;
}

export type WidgetId = string | number;

export enum WidgetState {
  Idle = "idle",
  Busy = "busy",
}

export abstract class BaseWidget implements WidgetOptions {
  label = "";
  static create(): BaseWidget { return new BaseWidget(); }
  async refresh(id: WidgetId): Promise<void> { await helper(id); }
  protected get caption(): string { return this.label; }
}

export class Widget extends BaseWidget {
  constructor(public label: string) { super(); }
  render(): string { return this.label; }
}

export const makeWidget = (label: string) => new Widget(label);
export let bindWidget = async (id: WidgetId) => id;
var legacyWidget = function (id: WidgetId) { return id; };
export { makeWidget as buildWidget, legacyWidget };
export default Widget;
