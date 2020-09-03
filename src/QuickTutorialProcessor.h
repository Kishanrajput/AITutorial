
#ifndef _QuickTutorialProcessor_h_
#define _QuickTutorialProcessor_h_

#include <JANA/JEventProcessor.h>
#include "../include/Model.h"
#include "../include/Tensor.h"

class QuickTutorialProcessor : public JEventProcessor {

    // Shared state (e.g. histograms, TTrees, TFiles) live
    // std::mutex m_mutex;
    
    
public:

	// Model model("../model.pb");
	
    QuickTutorialProcessor();
    virtual ~QuickTutorialProcessor() = default;

    void Init() override;
    void Process(const std::shared_ptr<const JEvent>& event) override;
    void Finish() override;

};


#endif // _QuickTutorialProcessor_h_

