import { createSignal, Show } from "solid-js";

export interface DropZoneProps {
  onFiles: (files: File[]) => void;
  busy: boolean;
  hasArchive: boolean;
}

export function DropZone(props: DropZoneProps) {
  const [dragging, setDragging] = createSignal(false);
  let input!: HTMLInputElement;

  const accept = (files: FileList | null) => {
    if (!files) return;
    const zips = Array.from(files).filter((file) =>
      file.name.toLowerCase().endsWith(".zip"),
    );
    if (zips.length > 0) props.onFiles(zips);
  };

  const openPicker = () => {
    if (!props.busy) input.click();
  };

  return (
    <div class="upload-area upload-area--archive">
      <div
        class="dropzone"
        classList={{ "dropzone--active": dragging(), "dropzone--busy": props.busy }}
        role="button"
        tabindex={props.busy ? -1 : 0}
        aria-disabled={props.busy}
        aria-label={props.hasArchive ? "別のXアーカイブを読み込む" : "XアーカイブのZIPを選ぶ"}
        aria-describedby="archive-help"
        aria-busy={props.busy}
        onDragOver={(event) => {
          event.preventDefault();
          if (!props.busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!props.busy) accept(event.dataTransfer?.files ?? null);
        }}
        onClick={openPicker}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openPicker();
          }
        }}
      >
        <input
          ref={input}
          type="file"
          accept=".zip,application/zip"
          multiple
          hidden
          onChange={(event) => {
            accept(event.currentTarget.files);
            event.currentTarget.value = "";
          }}
        />
        <div class="dropzone__icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" role="presentation">
            <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V14" />
          </svg>
        </div>
        <div class="dropzone__copy">
          <p class="dropzone__title">
            {props.hasArchive ? "別のXアーカイブを読み込む" : "XアーカイブのZIPを読み込む"}
          </p>
        </div>
        <Show when={props.busy}>
          <p class="dropzone__busy">読み込み中…</p>
        </Show>
      </div>
      <div class="archive-help archive-help--primary" id="archive-help">
        <p class="archive-help__title">Xアーカイブの入手方法</p>
        <ol class="archive-help__steps">
          <li>Xの「設定」→「アカウント」→「データとプライバシー」→「アーカイブの要求」でリクエストします。</li>
          <li>ダウンロードしたZIPを、この領域に読み込んでください。ファイルはブラウザ内で開かれ、サーバーへ送信されません。</li>
        </ol>
      </div>
    </div>
  );
}
