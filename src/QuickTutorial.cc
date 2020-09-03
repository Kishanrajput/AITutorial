

#include <JANA/JApplication.h>
#include <JANA/JFactoryGenerator.h>

#include "QuickTutorialProcessor.h"
#include "RandomSource.h"

extern "C" {
void InitPlugin(JApplication* app) {

    // This code is executed when the plugin is attached.
    // It should always call InitJANAPlugin(app) first, and then do one or more of:
    //   - Read configuration parameters
    //   - Register JFactoryGenerators
    //   - Register JEventProcessors
    //   - Register JEventSourceGenerators (or JEventSources directly)
    //   - Register JServices

    InitJANAPlugin(app);

    LOG << "Loading QuickTutorial" << LOG_END;
    app->Add(new QuickTutorialProcessor);
    // Add any additional components as needed
    app->Add(new RandomSource("random", app));
}
}

