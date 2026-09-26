export async function mountAgentUiAnnotation(): Promise<void> {
  const { init } = await import("agent-ui-annotation/vanilla");
  init({
    theme: "auto",
    outputLevel: "detailed",
    onBeforeAnnotationCreate: () => ({
      context: {
        route: window.location.pathname,
        title: document.title,
      },
    }),
  });
}
