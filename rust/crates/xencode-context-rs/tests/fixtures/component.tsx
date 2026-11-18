import React from "react";

/** A button with a label. */
export interface ButtonProps {
  label: string;
  disabled?: boolean;
}

export function Button(props: ButtonProps) {
  const onClick = () => props.label;
  return <button className="btn" onClick={onClick}>{props.label}</button>;
}

export class ButtonGroup extends React.Component<ButtonProps> {
  render() {
    return <div><Button label="ok" /></div>;
  }
}
