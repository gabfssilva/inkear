import SwiftUI
import UniformTypeIdentifiers

@main
struct InkearApp: App {
  @State private var model = AppModel()

  var body: some Scene {
    WindowGroup {
      RootView(model: model)
    }
  }
}

struct RootView: View {
  @Bindable var model: AppModel

  var body: some View {
    NavigationStack(path: $model.path) {
      LibraryView(model: model)
        .toolbar(.hidden, for: .navigationBar)
        .navigationDestination(for: AppModel.Route.self) { route in
          // Pushed screens get an opaque system background of their own, over the root's ground.
          destination(for: route)
            .toolbar(.hidden, for: .navigationBar)
            .themed()
        }
    }
    .themed()
    .fileImporter(
      isPresented: $model.isImporting, allowedContentTypes: [.pdf], allowsMultipleSelection: true
    ) { result in
      model.importDocuments(result)
    }
    .onOpenURL { url in
      model.importDocument(at: url)
    }
    #if DEBUG
      // UI tests cannot drive the system file picker, so they hand the app a PDF path instead;
      // on a device, a path relative to Documents, where `devicectl device copy to` puts the PDF.
      .task {
        if let path = ProcessInfo.processInfo.environment["INKEAR_IMPORT"] {
          model.importDocument(at: URL(filePath: path, relativeTo: .documentsDirectory).absoluteURL)
        }
      }
    #endif
    .alert("Error", isPresented: isAlertShown) {
      Button("OK") {}
    } message: {
      Text(verbatim: model.alert ?? "")
    }
  }

  private var isAlertShown: Binding<Bool> {
    Binding(
      get: { model.alert != nil },
      set: { if !$0 { model.dismissAlert() } })
  }

  @ViewBuilder
  private func destination(for route: AppModel.Route) -> some View {
    switch route {
    case .preparing(let id):
      PreparingView(model: model, id: id)
    case .ready(let id):
      ReadyView(model: model, id: id)
    case .player:
      PlayerView(model: model)
    case .models:
      ModelsView(model: model)
    }
  }
}
